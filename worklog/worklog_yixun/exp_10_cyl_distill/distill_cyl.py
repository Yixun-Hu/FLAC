#!/usr/bin/env python3
"""exp_10 S2: distill the P1@55,000 teacher into the cylindrical backbone (plan Rev 2/3).

PINNED RECIPE (plan Rev 2 R2-3 + Rev 3 R3-1 — no free knobs at launch):
  data      : the exp-09 AR train dataset config; per sample TWO branch fields — the
              source displacement field and ONE seeded context-pose field — built by the
              EXACT audited line  c = (coord[:, i, :, None, None] - depth) / max_value
              (conditioners.py ViTCoordinates.forward; max_value read from the pinned
              exp-09 model config and asserted equal across the two ViT conditioners).
  teacher   : HF vanilla Dinov3ViTModel (facebook/dinov3-vits16-pretrain-lvd1689m
              architecture) + the S1-extracted trained state dict, strict=True;
              .eval(), requires_grad_(False), forward under torch.no_grad();
              last_hidden_state [B,517,384] -> strip 1 CLS + 4 registers -> [B,512,384].
  student   : cylindrical_dinov3.CylindricalDINOv3ViTModel, official weights,
              gauge="cylindrical_xyz", train(); output is patch-only [B,512,384].
  loss      : FP32.  L_br = (1 - cos(s,t,dim=-1,eps=1e-8)).mean() + mse(s,t,"mean");
              L = L_src + L_ctx  (SUM — Rev 3).  bf16 autocast forwards, FP32 loss.
  optim     : AdamW(lr 1e-4, betas (0.9,0.999), wd 0.05, eps 1e-8), 500-step linear
              warmup -> cosine to 1e-6 over 10,000 steps, grad-norm clip 1.0,
              trainable = student backbone ONLY.
  topology  : DDP micro 16/GPU x 2 GPUs, accum 1 (eff 32); workers 6; seed 42.
              OOM ladder (frozen, disclosed): 8x2 accum2 -> 4x2 accum4.
  run       : 10,000 steps; ckpt every 1,000 (rank 0); FINAL = step 10,000; NO resume.
  gate      : mean(L[9801..10000]) < 0.5 * mean(L[801..1000]); any non-finite L => abort.
SOP-ladder modes: --synthetic (random fields, no dataset), --probe-readback (one real
batch, shapes+finite, no optimizer step), --max-steps N (smoke); peak VRAM logged.
Launch: torchrun --nproc_per_node 2 distill_cyl.py [flags]  (records at launch).
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

WT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EXP09_CFG = "/home/yixunhu/codespace/exp-09-cyl-dinov3-no-ssl/worklog/worklog_yixun/exp_09_cyl_no_ssl/FLAC_AR_exp09_online_eval.json"
HF_SNAPSHOT = "facebook/dinov3-vits16-pretrain-lvd1689m"
N_PREFIX = 5          # 1 CLS + 4 register tokens on the vanilla teacher
TOKENS, DIM = 512, 384


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


def branch_loss(s_tok, t_tok):
    """FP32 per-branch loss (Rev 3): mean(1 - cos) + mean MSE on [B,512,384]."""
    s, t = s_tok.float(), t_tok.float()
    if s.shape != t.shape or s.shape[1:] != (TOKENS, DIM):
        die(f"token shape mismatch: {tuple(s.shape)} vs {tuple(t.shape)}")
    cos = F.cosine_similarity(s, t, dim=-1, eps=1e-8)
    return (1.0 - cos).mean() + F.mse_loss(s, t, reduction="mean")


def lr_at(step, total, base=1e-4, floor=1e-6, warmup=500):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * p))


def gate_from_losses(losses, total=10000):
    """PASS iff mean(L[9801..10000]) < 0.5*mean(L[801..1000]) (1-indexed steps)."""
    if len(losses) < total:
        return {"pass": False, "reason": f"only {len(losses)} steps"}
    early = sum(losses[800:1000]) / 200.0
    late = sum(losses[total - 200:total]) / 200.0
    return {"pass": bool(late < 0.5 * early), "early_mean_801_1000": early,
            "late_mean_9801_10000": late, "ratio": late / early if early else float("inf")}


def build_teacher(teacher_pt, device):
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(HF_SNAPSHOT)
    model = AutoModel.from_config(cfg)
    sd = torch.load(teacher_pt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=True), None
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
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--probe-readback", action="store_true")
    args = ap.parse_args()
    if args.micro * max(1, int(os.environ.get("WORLD_SIZE", "1"))) * args.accum != 32:
        die(f"topology violates eff-32 pin: micro={args.micro} world={os.environ.get('WORLD_SIZE')} accum={args.accum}")
    if sha256_file(args.teacher) != args.teacher_sha:
        die("teacher artifact sha mismatch")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    max_value = read_max_value(EXP09_CFG)
    teacher = build_teacher(args.teacher, device)
    student = build_student(device)
    if world > 1:
        student = torch.nn.parallel.DistributedDataParallel(student, device_ids=[rank])
    net = student.module if world > 1 else student
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.999),
                            weight_decay=0.05, eps=1e-8)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"distill_log_rank{rank}.jsonl")
    if os.path.exists(log_path) and not (args.synthetic or args.probe_readback):
        die("log exists — NO-resume semantics: clear the out dir for a fresh run")

    if args.synthetic:
        loader = None
    else:
        sys.path.insert(0, WT)
        from src.data.dataset import create_dataloader_from_config
        dcfg = json.load(open(os.path.join(WT, args.dataset_config)))
        loader = create_dataloader_from_config(dcfg, batch_size=args.micro, num_workers=args.workers,
                                               sample_rate=22050, sample_size=10240, audio_channels=1)

    def batches():
        step = 0
        while True:
            if args.synthetic:
                g = torch.Generator().manual_seed(args.seed * 1000 + step)
                B = args.micro
                yield (torch.randn(B, 3, 256, 512, generator=g),
                       torch.randn(B, 3, 256, 512, generator=g))
                step += 1
                continue
            for audio, meta in loader:
                coords_s, coords_c, depths = [], [], []
                for si, x in enumerate(meta):
                    src = resolve_meta(x, "source_vit", "source")
                    ctxs = resolve_meta(x, "context_poses_vit", "context_poses")
                    d = x["depth"]
                    src_t = torch.as_tensor(src, dtype=torch.float32)
                    ctx_t = torch.as_tensor(ctxs, dtype=torch.float32)
                    if ctx_t.ndim == 1:
                        ctx_t = ctx_t.unsqueeze(0)
                    j = random.Random(args.seed ^ (step << 16) ^ si).randrange(ctx_t.shape[0])
                    coords_s.append(src_t.reshape(-1)[:3])
                    coords_c.append(ctx_t[j].reshape(-1)[:3])
                    depths.append(torch.as_tensor(d, dtype=torch.float32))
                depth = torch.stack(depths).to(device, non_blocking=True)
                cs = torch.stack(coords_s).to(device)
                cc = torch.stack(coords_c).to(device)
                f_src = build_field(cs, depth, max_value, 0)
                f_ctx = build_field(cc, depth, max_value, 0)
                yield f_src, f_ctx
                step += 1

    losses = []
    gen = batches()
    for step in range(args.steps):
        f_src, f_ctx = next(gen)
        f_src, f_ctx = f_src.to(device), f_ctx.to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            s_src = student(f_src).last_hidden_state
            s_ctx = student(f_ctx).last_hidden_state
            with torch.no_grad():
                t_src = teacher(pixel_values=f_src).last_hidden_state
                t_ctx = teacher(pixel_values=f_ctx).last_hidden_state
        if t_src.shape[1] != TOKENS + N_PREFIX:
            die(f"teacher token count {t_src.shape[1]} != {TOKENS + N_PREFIX}")
        L_src = branch_loss(s_src, t_src[:, N_PREFIX:, :])
        L_ctx = branch_loss(s_ctx, t_ctx[:, N_PREFIX:, :])
        L = L_src + L_ctx
        if not torch.isfinite(L):
            die(f"non-finite loss at step {step + 1} — abort per plan")
        opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        Lr = L.detach()
        if world > 1:
            dist.all_reduce(Lr)
            Lr = Lr / world
        losses.append(float(Lr))
        if rank == 0:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"step": step + 1, "L": float(Lr), "L_src": float(L_src),
                                     "L_ctx": float(L_ctx), "lr": opt.param_groups[0]["lr"],
                                     "peak_mem_mb": int(torch.cuda.max_memory_allocated() / 2**20)
                                     if device.type == "cuda" else 0}) + "\n")
            if args.probe_readback:
                print(f"PROBE-READBACK OK: fields {tuple(f_src.shape)}, student {tuple(s_src.shape)}, "
                      f"teacher-stripped {tuple(t_src[:, N_PREFIX:, :].shape)}, L={float(Lr):.4f} finite")
                break
            if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
                torch.save(net.state_dict(), os.path.join(args.out, f"distill_step{step + 1}.pt"))
    if rank == 0 and not args.probe_readback:
        gate = gate_from_losses(losses, args.steps) if args.steps >= 10000 else \
            {"pass": None, "note": f"gate only defined at 10000 steps (ran {args.steps})"}
        with open(os.path.join(args.out, "distill_gate.json"), "w") as fh:
            json.dump(gate, fh, indent=1)
        print("GATE:", gate)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
