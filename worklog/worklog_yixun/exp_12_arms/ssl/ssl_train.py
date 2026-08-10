"""exp_12 arm B -- native cylindrical DINO + iBOT + Gram SSL.

Adapts the official DINOv3 ViT-S/16 weights to the acoustic-geometry domain on the 243
TRAIN rooms only, then exports a backbone state dict that the C2 conditioning run loads in
place of `from_pretrained(...)` (see the `ssl_ckpt` key in conditioners.py).

  python ssl_train.py --out-dir outputs_FLAC/exp12B_ssl --gpu 0

Run it with PYTHONPATH=<cylindrical-dinov3>/src (package is never installed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

import ssl_data as D
from ssl_losses import DINOLoss, IBOTLoss, cosine_schedule, gram_loss, koleo_loss
from ssl_model import GramTeacher, SSLModel, ema_update, make_teacher

DATASET_ROOT = "/home/yixunhu/codespace/FLAC/AcousticRooms/"


def enable_grad_checkpointing(backbone) -> None:
    """Recompute each block in backward: ~3x activation-memory saving, same gradients."""
    for layer in backbone.layer:
        orig = layer.forward

        def wrapped(*args, _orig=orig, **kwargs):
            if torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _orig(*args, **kwargs)

        layer.forward = wrapped


def build_backbone(args):
    from cylindrical_dinov3 import CylindricalDINOv3ViTModel

    return CylindricalDINOv3ViTModel.from_pretrained(
        args.hf_model,
        gauge="cylindrical_xyz",
        azimuth_mode=args.azimuth_mode,
        prefix_mode=args.prefix_mode,
        attn_implementation="eager",
    )


def param_groups(model: torch.nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-local", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-end", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--wd-end", type=float, default=0.2)
    ap.add_argument("--momentum", type=float, default=0.994)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--teacher-temp-end", type=float, default=0.07)
    ap.add_argument("--teacher-temp-warmup", type=int, default=5000)
    ap.add_argument("--out-dim", type=int, default=8192)
    ap.add_argument("--ibot-out-dim", type=int, default=4096)
    ap.add_argument("--w-dino", type=float, default=1.0)
    ap.add_argument("--w-ibot", type=float, default=1.0)
    ap.add_argument("--w-gram", type=float, default=1.0)
    ap.add_argument("--w-koleo", type=float, default=0.1)
    ap.add_argument("--gram-start", type=int, default=10000)
    ap.add_argument("--gram-refresh", type=int, default=10000)
    ap.add_argument("--freeze-last-layer", type=int, default=1000)
    ap.add_argument("--clip-grad", type=float, default=3.0)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--azimuth-mode", default="lowband")
    ap.add_argument("--prefix-mode", default="m0_registers")
    ap.add_argument("--hf-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--manifest", default="data/AR/train.json")
    ap.add_argument("--forbidden-manifest", default="data/AR/unseen_eval.json")
    ap.add_argument("--index-cache", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ---- corpus (train rooms only; the guard REFUSES on any held-out room) -------------
    cache = args.index_cache or os.path.join(args.out_dir, "ssl_index.json")
    if os.path.exists(cache):
        index = D.load_index(cache)
        print(f"[data] loaded index: {len(index)} rooms from {cache}", flush=True)
    else:
        t0 = time.time()
        index = D.build_index(DATASET_ROOT, args.manifest, args.forbidden_manifest)
        D.save_index(index, cache)
        print(f"[data] built index: {len(index)} rooms in {time.time()-t0:.1f}s -> {cache}", flush=True)
    forbidden = {
        (s, r) for s, rooms in json.load(open(args.forbidden_manifest)).items() for r in rooms
    }
    ds = D.RoomViewDataset(
        index, DATASET_ROOT, n_local=args.n_local, forbidden_rooms=forbidden, seed=args.seed
    )
    sampler = RandomSampler(ds, replacement=True, num_samples=args.batch_size * args.steps)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # ---- models ------------------------------------------------------------------------
    student = SSLModel(build_backbone(args), args.out_dim, args.ibot_out_dim).to(dev)
    teacher = make_teacher(student).to(dev)
    enable_grad_checkpointing(student.backbone)
    gram = GramTeacher()

    dino_loss = DINOLoss(args.out_dim).to(dev)
    ibot_loss = IBOTLoss(args.ibot_out_dim).to(dev)
    opt = torch.optim.AdamW(param_groups(student, args.wd), lr=args.lr, betas=(0.9, 0.999))

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        blob = torch.load(args.resume, map_location="cpu")
        student.load_state_dict(blob["student"])
        teacher.load_state_dict(blob["teacher"])
        opt.load_state_dict(blob["opt"])
        dino_loss.load_state_dict(blob["dino_loss"])
        ibot_loss.load_state_dict(blob["ibot_loss"])
        start_step = blob["step"]
        if blob.get("gram") is not None:
            gram.refresh(teacher)
        print(f"[resume] step {start_step} from {args.resume}", flush=True)

    logf = open(os.path.join(args.out_dir, "ssl_log.jsonl"), "a")
    json.dump({"args": vars(args), "n_rooms": len(index)}, open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)

    def save(step: int, final: bool = False) -> None:
        torch.save(
            {
                "step": step,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "opt": opt.state_dict(),
                "dino_loss": dino_loss.state_dict(),
                "ibot_loss": ibot_loss.state_dict(),
                "gram": gram.ready,
                "args": vars(args),
            },
            os.path.join(args.out_dir, "last.pt"),
        )
        name = "backbone_final.pt" if final else f"backbone_step{step}.pt"
        torch.save(
            {
                "step": step,
                "backbone": {k: v.cpu() for k, v in teacher.backbone.state_dict().items()},
                "azimuth_mode": args.azimuth_mode,
                "prefix_mode": args.prefix_mode,
                "hf_model": args.hf_model,
            },
            os.path.join(args.out_dir, name),
        )

    # ---- train ---------------------------------------------------------------------------
    student.train()
    t0, step = time.time(), start_step
    for batch in dl:
        if step >= args.steps:
            break
        lr = cosine_schedule(step, args.steps, args.lr, args.lr_end, args.warmup)
        wd = cosine_schedule(step, args.steps, args.wd, args.wd_end)
        mom = cosine_schedule(step, args.steps, args.momentum, 1.0)
        ttemp = args.teacher_temp + (args.teacher_temp_end - args.teacher_temp) * min(
            step / max(args.teacher_temp_warmup, 1), 1.0
        )
        for g in opt.param_groups:
            g["lr"] = lr
            if g["weight_decay"] > 0:
                g["weight_decay"] = wd

        g_views = batch["globals"].to(dev, non_blocking=True)          # [B, 2, 3, H, W]
        l_views = batch["locals"].to(dev, non_blocking=True)           # [B, L, 3, h, w]
        masks = batch["masks"].to(dev, non_blocking=True)              # [B, 2, N] bool
        B = g_views.shape[0]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            with torch.no_grad():
                t_out = [teacher(g_views[:, i]) for i in range(2)]
                t_glob = [teacher.dino_head(o["pooled"]) for o in t_out]

            s_out = [student(g_views[:, i], bool_masked_pos=masks[:, i]) for i in range(2)]
            s_glob = [student.dino_head(o["pooled"]) for o in s_out]
            for i in range(l_views.shape[1]):
                s_glob.append(student.dino_head(student(l_views[:, i])["pooled"]))

            l_dino = dino_loss(s_glob, t_glob, ttemp)

            # iBOT: student's mask-token positions vs teacher's real features there.
            sp, tp = [], []
            for i in range(2):
                m = masks[:, i]
                if m.any():
                    sp.append(student.ibot_head(s_out[i]["patch"][m]))
                    tp.append(teacher.ibot_head(t_out[i]["patch"][m]))
            if sp:
                sp_c, tp_c = torch.cat(sp), torch.cat(tp)
                l_ibot = ibot_loss(sp_c, tp_c, ttemp)
            else:
                sp_c = tp_c = None
                l_ibot = g_views.new_zeros(())

            l_koleo = sum(koleo_loss(o["pooled"]) for o in s_out) / 2.0

            if args.w_gram > 0 and step >= args.gram_start:
                if not gram.ready or (step - args.gram_start) % max(args.gram_refresh, 1) == 0:
                    gram.refresh(teacher)
                l_gram = sum(
                    gram_loss(s_out[i]["patch"], gram.patches(g_views[:, i])) for i in range(2)
                ) / 2.0
            else:
                l_gram = g_views.new_zeros(())

            loss = (
                args.w_dino * l_dino
                + args.w_ibot * l_ibot
                + args.w_gram * l_gram
                + args.w_koleo * l_koleo
            )

        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at step {step}; skipping", flush=True)
            opt.zero_grad(set_to_none=True)
            step += 1
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step < args.freeze_last_layer:                 # DINO stability trick
            for h in (student.dino_head, student.ibot_head):
                for p in h.last_layer.parameters():
                    p.grad = None
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad)
        opt.step()

        with torch.no_grad():
            ema_update(teacher, student, mom)
            dino_loss.update_center(t_glob)
            if tp_c is not None:
                ibot_loss.update_center(tp_c)

        step += 1
        if step % args.log_every == 0:
            rec = {
                "step": step,
                "loss": float(loss),
                "dino": float(l_dino),
                "ibot": float(l_ibot),
                "gram": float(l_gram),
                "koleo": float(l_koleo),
                "lr": lr,
                "mom": mom,
                "ttemp": ttemp,
                "steps_per_s": step and (step - start_step) / (time.time() - t0),
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(
                f"step {step}/{args.steps} loss {rec['loss']:.4f} "
                f"(dino {rec['dino']:.4f} ibot {rec['ibot']:.4f} gram {rec['gram']:.4f} "
                f"koleo {rec['koleo']:.4f}) lr {lr:.2e} {rec['steps_per_s']:.3f} it/s",
                flush=True,
            )
        if step % args.ckpt_every == 0:
            save(step)

    save(step, final=True)
    print(f"[done] {step} steps in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
