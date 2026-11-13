"""Synthetic only (no AcousticRooms file): 33 random furnished shoebox rooms in the TRUE frame,
6 receivers, source counts like the frozen real selection; evaluate the registered gate."""
import importlib.util, json, sys, time
import numpy as np, torch
spec = importlib.util.spec_from_file_location("t", "/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_exp43_frame_audit.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
fa = t.fa
torch.set_num_threads(12)
seed = int(sys.argv[1])
rng = np.random.default_rng(seed)
NSRC = [49, 10, 50, 50, 99, 10, 10, 10, 10, 10, 10, 10, 10, 100, 75, 10, 75, 10, 32, 47, 36, 24, 10, 49, 45, 39, 10, 50, 50, 50, 10, 50, 50]
CATS = ["Ap"] * 4 + ["Au"] + ["Ba"] * 4 + ["Be"] * 4 + ["Ca"] + ["Li"] * 3 + ["Lv"] * 4 + ["Me"] * 4 + ["Of"] * 4 + ["Re"] * 4

def free_point(size, boxes, margin):
    while True:
        p = margin + rng.uniform(0, 1, 3) * (size - 2 * margin)
        if not any(np.all(p > np.asarray(lo) - margin) and np.all(p < np.asarray(hi) + margin) for lo, hi in boxes):
            return p

rooms = []
t0 = time.time()
for n in range(33):
    size = np.array([rng.uniform(3, 12), rng.uniform(3, 9), rng.uniform(2.4, 4.0)])
    boxes = []
    for _ in range(rng.integers(3, 9)):
        c = rng.uniform(0, 1, 3) * size
        e = np.array([rng.uniform(0.3, 1.8), rng.uniform(0.3, 1.8), rng.uniform(0.4, size[2])])
        lo = np.maximum(c - e / 2, 0); lo[2] = 0.0
        hi = np.minimum(c + e / 2, size); hi[2] = min(e[2], size[2])
        boxes.append((tuple(lo), tuple(hi)))
    sp = dict(room_id=f"R{n}", category=CATS[n], size=tuple(size), boxes=boxes)
    sp["receivers"] = [tuple(free_point(size, boxes, 0.3)) for _ in range(6)]
    sp["sources"] = [tuple(free_point(size, boxes, 0.3)) for _ in range(NSRC[n])]
    rs = t.room_struct(sp, 0, 0)
    rooms.append({"room_id": sp["room_id"], "category": sp["category"], "scores": fa.score_room(rs)})
g = fa.evaluate_gate(rooms)
per_room_A = [[fa.hyp_label(h) for h in fa.argmax_set(fa.room_fractions(r["scores"])["A"])] for r in rooms]
per_room_B = [[fa.hyp_label(h) for h in fa.argmax_set(fa.room_fractions(r["scores"])["B"])] for r in rooms]
out = {"seed": seed, "verdict": g["verdict"], "reasons": g["reasons"], "full_A": g["full"]["A"]["argmax"],
       "full_B": g["full"]["B"]["argmax"], "boot_pass": g["bootstrap"]["n_pass"],
       "boot_hist_A": g["bootstrap"]["argmax_histogram_A"], "boot_hist_B": g["bootstrap"]["argmax_histogram_B"],
       "loco_A": {c: v["A"]["argmax"] for c, v in g["loco"].items()},
       "rooms_A_exact": sum(a == ["I:k=0"] for a in per_room_A), "rooms_B_exact": sum(b == ["I:k=0"] for b in per_room_B),
       "per_room_A": per_room_A, "A_curve_near0": [round(x, 5) for x in np.roll(np.asarray(g["full"]["A"]["curve"])[0], 8)[:17]],
       "secs": round(time.time() - t0, 1)}
print(json.dumps(out))
