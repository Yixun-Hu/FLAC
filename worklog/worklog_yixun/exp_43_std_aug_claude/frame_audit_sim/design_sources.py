"""Design-time only: choose per-room source layouts for which Score A has a unique maximum at the
true frame. Candidates are random free-space points; selection = fully visible sources whose
small-shift losses cover every wrong hypothesis (margin >= 2), plus a few deeply occluded ones.

Usage, from any directory: python design_sources.py --out designed_sources.json   (one line per room
to stdout; --out is required and never overwritten).
"""
import argparse, importlib.util, os, sys, json
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[4]          # .../worklog/worklog_yixun/exp_43_std_aug_claude/frame_audit_sim
AUDIT_TESTS = REPO / "src" / "tests" / "test_exp43_frame_audit.py"


def load_audit_test_module():
    """The frame-audit test module (its ``fa`` is the audit tool), resolved from this file's location."""
    if not AUDIT_TESTS.is_file():
        raise FileNotFoundError(f"frame-audit test module not found at {AUDIT_TESTS}")
    spec = importlib.util.spec_from_file_location("t", AUDIT_TESTS)
    t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
    return t


W = 512


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="JSON file to write (must not exist; its directory must)")
    args = ap.parse_args(argv)
    out_path = os.path.abspath(args.out)
    if os.path.exists(out_path) or not os.path.isdir(os.path.dirname(out_path)):
        why = "refusing to overwrite" if os.path.exists(out_path) else "no such directory for"
        print(f"design_sources: {why} {out_path}", file=sys.stderr)
        return 2
    t = load_audit_test_module()
    fa = t.fa

    def pair_vectors(sp, cands):
        recs = sp["receivers"]
        C = np.zeros((len(recs), len(cands), 2 * W), dtype=np.int64)
        for ri, r in enumerate(recs):
            d = t.render(sp, r, 0, 0)
            dflat = torch.from_numpy(d.reshape(-1)); valid = dflat > 0
            u = np.asarray(cands) - np.asarray(r)
            for si in range(len(cands)):
                C[ri, si] = fa._score_a_receiver(u[si:si + 1], dflat, valid).ravel()
        return C

    def free_candidates(sp, n, seed):
        rng = np.random.default_rng(seed); out = []
        size = np.asarray(sp["size"])
        while len(out) < n:
            p = 0.2 + rng.uniform(0, 1, 3) * (size - 0.4)
            if any(np.all(p > np.asarray(lo) - 0.2) and np.all(p < np.asarray(hi) + 0.2) for lo, hi in sp["boxes"]):
                continue
            if min(np.linalg.norm(p - np.asarray(r)) for r in sp["receivers"]) < 0.5:
                continue
            out.append(tuple(np.round(p, 3)))
        return out

    result = {}
    for n, sp in enumerate(t.ROOMS):
        cands = free_candidates(sp, 1500, 9100 + n)
        C = pair_vectors(sp, cands)                     # [R, S, H]
        nR = C.shape[0]
        cs = C.sum(axis=0)                              # per source, summed over receivers [S, H]
        vis = [i for i in range(len(cands)) if cs[i, 0] == nR]
        occ = [i for i in range(len(cands)) if cs[i, 0] <= nR - 2]
        # deeply occluded: no gain anywhere within +-8 columns (both S) and never above its truth by > 0 near 0
        near = [k for k in range(-8, 9) if k != 0]
        deep = [i for i in occ if all(cs[i, k % W] <= cs[i, 0] for k in near)]
        chosen = []
        loss = np.zeros(2 * W, dtype=np.int64)          # accumulated margin sum_s (c_s(0) - c_s(h))
        target = 3
        pool = list(vis)
        while True:
            need = np.flatnonzero(loss < target); need = need[need != 0]
            if len(need) == 0 or not pool:
                break
            gains = [(np.sum(np.minimum(cs[i, 0] - cs[i, need], target)), i) for i in pool]
            g, best = max(gains)
            if g <= 0:
                break
            chosen.append(best); pool.remove(best)
            loss += cs[best, 0] - cs[best]
        # add up to 4 deep-occluded sources if margin stays >= 1 everywhere
        added = 0
        for i in deep:
            if added == 4: break
            trial = loss + cs[i, 0] - cs[i]
            trial[0] = 99
            if trial.min() >= 2:
                chosen.append(i); loss = trial; loss[0] = 0; added += 1
        # pad with more visible sources to at least 14 sources
        for i in pool:
            if len(chosen) >= 20: break
            if i not in chosen:
                chosen.append(i); loss += cs[i, 0] - cs[i]
        tot = cs[chosen].sum(axis=0)
        am = fa.argmax_set(tot.astype(float))
        mm = np.delete(tot, 0).max()
        print(sp["room_id"], "n_cand", len(cands), "vis", len(vis), "deep", len(deep), "chosen", len(chosen),
              "occluded added", added, "argmax", [fa.hyp_label(h) for h in am], "margin", int(tot[0] - mm),
              "n_blocked_pairs", int(sum(nR - cs[i, 0] for i in chosen)), "weak h", [fa.hyp_label(h) for h in np.flatnonzero(tot[0] - tot < 3) if h != 0][:10])
        result[sp["room_id"]] = [list(cands[i]) for i in sorted(chosen)]
    try:
        with open(out_path, "x") as fh:                # exclusive: never overwrites
            json.dump(result, fh, indent=1)
    except FileExistsError:
        print(f"design_sources: refusing to overwrite {out_path}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
