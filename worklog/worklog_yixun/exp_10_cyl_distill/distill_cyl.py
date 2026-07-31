#!/usr/bin/env python3
"""exp_10 S2: distill the P1@55,000 teacher into the cylindrical backbone.

PINNED RECIPE (plan Rev 2 R2-3 + Rev 3 R3-1; code-r1 remediations). Modes:
  REAL run       : every pin FAIL-CLOSED (see real_run_gates) — frozen topology set,
                   steps=10000, seed=42, canonical S1 teacher (manifest-authenticated),
                   pinned dataset + config identity hashes, reviewed-code gates
                   (EXP10 worktree HEAD + tracked-clean; package-proper pin), fresh out
                   dir (log/ckpt/gate refusal), rc = 0 ONLY if the loss gate PASSES.
  --probe-readback: one real batch, forward + loss + finite/shape report ONLY (NO
                   backward/step), all ranks break together; REQUIRES an out dir
                   suffixed `_probe`; never writes checkpoints.
  --synthetic    : random fields, optimizer ON (ladder step for throughput/VRAM);
                   REQUIRES an out dir suffixed `_synthetic`; never writes checkpoints.
Gradient accumulation is REAL: per optimizer step, `accum` micro-batches each contribute
L_micro/accum via backward (DDP no_sync on non-final micros); effective batch =
micro x world x accum = 32 always. L_micro = L_src + L_ctx (SUM, Rev 3), FP32.
"""
import argparse
import contextlib
import glob
import hashlib
import json
import math
import os
import random
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

WT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MAIN = "/home/yixunhu/codespace/cylindrical-dinov3"
EXP09_CFG = "/home/yixunhu/codespace/exp-09-cyl-dinov3-no-ssl/worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp09_online_eval.json"
EXP09_CFG_SHA = "86f5e2bedde28a323e3b159d8a7ea93cb34e1bdea86d06c8f45236aa3f3b3bfa"
TRAIN_MANIFEST_REL = "data/AR/train.json"
TRAIN_MANIFEST_SHA = "aa4e52d616fc42e88d5e4952c7e7ff266347615a60f93a0590b707f5eeaead03"
DATASET_CFG_PIN = "src/configs/dataset_configs/AR/train/acousticroom_train.json"
PACKAGE_PIN = "301731b5540a22a6d42ec8926e53379854bf4f97"
CANON_TEACHER = os.path.join(WT, "outputs_FLAC", "exp10_teacher", "teacher_vit_p1s55000.pt")
CANON_MANIFEST = os.path.join(WT, "outputs_FLAC", "exp10_teacher", "teacher_manifest.json")
HF_SNAPSHOT = "facebook/dinov3-vits16-pretrain-lvd1689m"
N_PREFIX = 5          # 1 CLS + 4 register tokens on the vanilla teacher
TOKENS, DIM = 512, 384
FROZEN_TOPOLOGIES = {(16, 1), (8, 2), (4, 4)}   # (micro, accum) at world=2 — eff 32
STEPS_PIN, SEED_PIN, CKPT_EVERY_PIN = 10000, 42, 1000


def die(msg):
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(f"REFUSE: {msg}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_max_value(cfg_path):
    cfg = json.load(open(cfg_path))
    vals = [c["config"]["max_value"] for c in cfg["model"]["conditioning"]["configs"]
            if c.get("type") == "ViTCoordinates"]
    if len(vals) != 2 or vals[0] != vals[1]:
        die(f"ViT max_value not unique across the two conditioners: {vals}")
    return float(vals[0])


def resolve_meta(x, key, fallback):
    """MultiConditioner two-step key resolution (conditioners.py:377-386), fail-closed."""
    if key in x:
        return x[key]
    if fallback in x:
        return x[fallback]
    die(f"metadata lacks both {key!r} and {fallback!r}")


def build_field(coord, depth, max_value, i):
    """EXACT port of the audited line: (coord[:, i, :, None, None] - depth) / max_value."""
    if coord.ndim == 2:
        coord = coord.unsqueeze(1)
    return (coord[:, i, :, None, None] - depth) / max_value


def strip_prefix(t_tok):
    """Teacher last_hidden_state [B, 517, 384] -> patch-only [B, 512, 384]; refuse else."""
    if t_tok.shape[1] != TOKENS + N_PREFIX or t_tok.shape[2] != DIM:
        die(f"teacher token shape {tuple(t_tok.shape)} != [B, {TOKENS + N_PREFIX}, {DIM}]")
    return t_tok[:, N_PREFIX:, :]


def branch_loss(s_tok, t_tok):
    """FP32 per-branch loss (Rev 3): mean(1 - cos(eps=1e-8)) + mean MSE on [B,512,384]."""
    s, t = s_tok.float(), t_tok.float()
    if s.shape != t.shape or s.shape[1:] != (TOKENS, DIM):
        die(f"token shape mismatch: {tuple(s.shape)} vs {tuple(t.shape)}")
    cos = F.cosine_similarity(s, t, dim=-1, eps=1e-8)
    return (1.0 - cos).mean() + F.mse_loss(s, t, reduction="mean")


def total_loss(s_src, t_src_stripped, s_ctx, t_ctx_stripped):
    """The load-bearing Rev 3 combination: L = L_src + L_ctx (SUM, never mean)."""
    L_src = branch_loss(s_src, t_src_stripped)
    L_ctx = branch_loss(s_ctx, t_ctx_stripped)
    return L_src + L_ctx, L_src, L_ctx


def lr_at(step, total, base=1e-4, floor=1e-6, warmup=500):
    """Executed steps 0..total-1. Warmup ramps to base AT step warmup-1; cosine spans
    steps warmup..total-1 with p in (0, 1], reaching EXACTLY floor at the final
    executed step (code-r1 #6: no base repeat, exact endpoint)."""
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - (warmup - 1)) / (total - warmup)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * p))


def gate_from_losses(losses, total=STEPS_PIN):
    """PASS iff mean(L[9801..10000]) < 0.5*mean(L[801..1000]) (1-indexed steps)."""
    if len(losses) < total:
        return {"pass": False, "reason": f"only {len(losses)} steps"}
    early = sum(losses[800:1000]) / 200.0
    late = sum(losses[total - 200:total]) / 200.0
    return {"pass": bool(late < 0.5 * early), "early_mean_801_1000": early,
            "late_mean_9801_10000": late, "ratio": late / early if early else float("inf")}


def git_out(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if r.returncode != 0:
        die(f"git {' '.join(args)} rc={r.returncode} in {repo} — refusing blind")
    return r.stdout.strip()


def real_run_gates(args, world):
    """Every fail-closed identity/pin gate for a REAL run (code-r1 #3/#4)."""
    if args.steps != STEPS_PIN or args.seed != SEED_PIN or args.ckpt_every != CKPT_EVERY_PIN:
        die(f"real run pins violated: steps={args.steps} seed={args.seed} ckpt_every={args.ckpt_every}")
    if world != 2 or (args.micro, args.accum) not in FROZEN_TOPOLOGIES:
        die(f"topology not in the frozen set: world={world} micro={args.micro} accum={args.accum}")
    if os.path.realpath(args.teacher) != os.path.realpath(CANON_TEACHER):
        die("real run must use the canonical S1 teacher artifact")
    if not os.path.isfile(CANON_MANIFEST):
        die("S1 teacher manifest missing")
    man = json.load(open(CANON_MANIFEST))
    actual = sha256_file(args.teacher)
    if actual != man.get("output_sha256") or actual != args.teacher_sha:
        die("teacher sha does not match BOTH the S1 manifest and the CLI pin")
    if os.path.realpath(os.path.join(WT, args.dataset_config)) != \
            os.path.realpath(os.path.join(WT, DATASET_CFG_PIN)):
        die("real run must use the pinned dataset config (r2 #1)")
    dcfg = json.load(open(os.path.join(WT, DATASET_CFG_PIN)))
    mpath = dcfg["datasets"][0].get("json_file_path", "")
    if os.path.realpath(os.path.join(WT, mpath)) != os.path.realpath(os.path.join(WT, TRAIN_MANIFEST_REL)):
        die(f"pinned dataset config points at {mpath!r}, not the hashed manifest (r2 #1)")
    if sha256_file(os.path.join(WT, TRAIN_MANIFEST_REL)) != TRAIN_MANIFEST_SHA:
        die("train.json manifest hash drift")
    if sha256_file(EXP09_CFG) != EXP09_CFG_SHA:
        die("exp-09 model config identity drift (max_value source)")
    exp = os.environ.get("EXPECT_EXP10_SHA")
    if not exp:
        die("EXPECT_EXP10_SHA required for a real run")
    if git_out(WT, "rev-parse", "HEAD") != exp:
        die("worktree HEAD != EXPECT_EXP10_SHA")
    if git_out(WT, "status", "--porcelain", "-uno"):
        die("worktree tracked-dirty")
    if git_out(MAIN, "log", "--format=%H", "-1", "--", "src/cylindrical_dinov3") != PACKAGE_PIN:
        die("cylindrical package-proper drifted from the pinned commit")
    for pat in ("distill_log.jsonl", "distill_step*.pt", "distill_gate.json"):
        if glob.glob(os.path.join(args.out, pat)):
            die(f"NO-resume: {pat} present in {args.out} — clear the dir for a fresh run")


def build_teacher(teacher_pt, device):
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(HF_SNAPSHOT)
    model = AutoModel.from_config(cfg)
    sd = torch.load(teacher_pt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_student(device):
    from cylindrical_dinov3 import CylindricalDINOv3ViTModel
    m = CylindricalDINOv3ViTModel.from_pretrained(HF_SNAPSHOT, gauge="cylindrical_xyz")
    return m.to(device).train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher-sha", required=True)
    ap.add_argument("--dataset-config", default="src/configs/dataset_configs/AR/train/acousticroom_train.json")
    ap.add_argument("--out", default=os.path.join(WT, "outputs_FLAC", "exp10_distill"))
    ap.add_argument("--steps", type=int, default=STEPS_PIN)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED_PIN)
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY_PIN)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--probe-readback", action="store_true")
    args = ap.parse_args()
    if args.synthetic and args.probe_readback:
        die("--synthetic and --probe-readback are mutually exclusive")
    probe_mode = args.synthetic or args.probe_readback
    if args.probe_readback and not args.out.rstrip("/").endswith("_probe"):
        die("--probe-readback requires an out dir suffixed _probe")
    if args.synthetic and not args.out.rstrip("/").endswith("_synthetic"):
        die("--synthetic requires an out dir suffixed _synthetic")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not probe_mode:
        real_run_gates(args, world)
    else:
        if args.micro * world * args.accum != 32:
            die(f"probe topology violates eff-32: micro={args.micro} world={world} accum={args.accum}")
        if sha256_file(args.teacher) != args.teacher_sha:
            die("teacher sha mismatch")
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    max_value = read_max_value(EXP09_CFG)
    teacher = build_teacher(args.teacher, device)
    student = build_student(device)
    if world > 1:
        # find_unused_parameters=True: the ViT mask_token never participates in these
        # forwards (ladder-2 DDP failure, disclosed); this mirrors the codebase's own
        # training strategy 'ddp_find_unused_parameters_true' (exp_06 C2 / P1 launches).
        student = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[rank], find_unused_parameters=True)
    net = student.module if world > 1 else student
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.999),
                            weight_decay=0.05, eps=1e-8)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "distill_log.jsonl")
    # r2 #2: EVERY rank checks (the file is on shared storage) so no rank proceeds alone.
    if os.path.exists(log_path):
        die("distill_log.jsonl exists — NO-resume semantics")

    if not args.synthetic:
        sys.path.insert(0, WT)
        from src.data.dataset import create_dataloader_from_config
        dcfg = json.load(open(os.path.join(WT, args.dataset_config)))
        loader = create_dataloader_from_config(dcfg, batch_size=args.micro, num_workers=args.workers,
                                               sample_rate=22050, sample_size=10240, audio_channels=1)

    def micro_batches():
        mstep = 0
        while True:
            if args.synthetic:
                g = torch.Generator().manual_seed(args.seed * 1000 + mstep)
                B = args.micro
                yield (torch.randn(B, 3, 256, 512, generator=g),
                       torch.randn(B, 3, 256, 512, generator=g))
                mstep += 1
                continue
            for _audio, meta in loader:
                coords_s, coords_c, depths = [], [], []
                for si, x in enumerate(meta):
                    src = resolve_meta(x, "source_vit", "source")
                    ctxs = resolve_meta(x, "context_poses_vit", "context_poses")
                    src_t = torch.as_tensor(src, dtype=torch.float32)
                    ctx_t = torch.as_tensor(ctxs, dtype=torch.float32)
                    if ctx_t.ndim == 1:
                        ctx_t = ctx_t.unsqueeze(0)
                    j = random.Random(args.seed ^ (mstep << 16) ^ si).randrange(ctx_t.shape[0])
                    coords_s.append(src_t.reshape(-1)[:3])
                    coords_c.append(ctx_t[j].reshape(-1)[:3])
                    depths.append(torch.as_tensor(x["depth"], dtype=torch.float32))
                depth = torch.stack(depths).to(device, non_blocking=True)
                f_src = build_field(torch.stack(coords_s).to(device), depth, max_value, 0)
                f_ctx = build_field(torch.stack(coords_c).to(device), depth, max_value, 0)
                yield f_src, f_ctx
                mstep += 1

    def forward_losses(f_src, f_ctx):
        # ONE concatenated forward per net per micro-batch: the ViT is per-sample
        # independent (no cross-sample ops), so cat->split is mathematically identical
        # to two forwards and avoids DDP multi-forward reducer hazards (ladder-2 fix).
        B = f_src.shape[0]
        f_cat = torch.cat([f_src, f_ctx], dim=0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            s_cat = student(f_cat).last_hidden_state
            with torch.no_grad():
                t_cat = teacher(pixel_values=f_cat).last_hidden_state
        t_p = strip_prefix(t_cat)
        return total_loss(s_cat[:B], t_p[:B], s_cat[B:], t_p[B:])

    gen = micro_batches()
    if args.probe_readback:
        f_src, f_ctx = next(gen)
        f_src, f_ctx = f_src.to(device), f_ctx.to(device)
        with torch.no_grad():        # NO backward, NO optimizer step (code-r1 #2)
            L, L_src, L_ctx = forward_losses(f_src, f_ctx)
        if not torch.isfinite(L):
            die("probe-readback loss non-finite")
        if rank == 0:
            print(f"PROBE-READBACK OK: fields {tuple(f_src.shape)}, L={float(L):.4f} "
                  f"(src {float(L_src):.4f} + ctx {float(L_ctx):.4f}) finite; NO optimizer step")
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    losses = []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps)
        opt.zero_grad(set_to_none=True)
        step_L = step_src = step_ctx = 0.0
        for m in range(args.accum):
            f_src, f_ctx = next(gen)
            f_src, f_ctx = f_src.to(device), f_ctx.to(device)
            sync_ctx = student.no_sync() if (world > 1 and m < args.accum - 1) else contextlib.nullcontext()
            with sync_ctx:
                L, L_src, L_ctx = forward_losses(f_src, f_ctx)
                if not torch.isfinite(L):
                    die(f"non-finite loss at step {step + 1} micro {m + 1} — abort per plan")
                (L / args.accum).backward()
            step_L += float(L.detach()) / args.accum
            step_src += float(L_src.detach()) / args.accum
            step_ctx += float(L_ctx.detach()) / args.accum
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        Lr = torch.tensor(step_L, device=device)
        if world > 1:
            dist.all_reduce(Lr)
            Lr = Lr / world
        losses.append(float(Lr))
        if rank == 0:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"step": step + 1, "L": float(Lr), "L_src": step_src,
                                     "L_ctx": step_ctx, "lr": opt.param_groups[0]["lr"],
                                     "peak_mem_mb": int(torch.cuda.max_memory_allocated() / 2**20)
                                     if device.type == "cuda" else 0}) + "\n")
            if not args.synthetic and ((step + 1) % args.ckpt_every == 0 or step + 1 == args.steps):
                torch.save(net.state_dict(), os.path.join(args.out, f"distill_step{step + 1}.pt"))
    rc = 0
    if rank == 0:
        if args.synthetic:
            print(f"SYNTHETIC RUN DONE ({args.steps} steps; no ckpts, no gate)")
        else:
            gate = gate_from_losses(losses, STEPS_PIN)
            with open(os.path.join(args.out, "distill_gate.json"), "w") as fh:
                json.dump(gate, fh, indent=1)
            print("GATE:", gate)
            rc = 0 if gate.get("pass") is True else 1   # gate FAIL => nonzero (code-r1 #4)
    if world > 1:
        rc_t = torch.tensor(rc, device=device)
        dist.broadcast(rc_t, src=0)
        rc = int(rc_t)
        dist.destroy_process_group()
    sys.exit(rc)


if __name__ == "__main__":
    main()
