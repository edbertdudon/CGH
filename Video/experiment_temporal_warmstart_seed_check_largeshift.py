"""
Audit follow-up (10.9): 10.5's steady-pan warm-start table shows a
3-seed range for the small-shift full-resolution case (~0.6% of frame,
7.67x-9.33x) but the large-shift full-resolution case (~12% of frame,
~6.0x) is still a single-seed number sitting in the same table --
genuinely easy to miss next to a properly-checked one. This closes that
gap the same way the small-shift case was closed
(experiment_temporal_warmstart_seed_check.py), just at the large-shift
magnitude (550px, ~11.7% of a 4700px-wide frame) instead of the small one.

Run on your 3060 (expect ~10 minutes):
    python3 experiment_temporal_warmstart_seed_check_largeshift.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch
from experiment_temporal_warmstart import check_patch_applied, first_crossing
from experiment_temporal_warmstart_padcrop import make_frame_sequence_padcrop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_FRAMES = 4
N_ITERS_BUDGET = 40
QUALITY_THRESHOLD_MARGIN_DB = 0.5
SHIFT_PX = 550   # ~11.7% of frame width -- the large-shift case from _fullres.py, never seed-checked
BASE_SEEDS = [0, 1, 2]


def run_sequence_seeded(frames, warm_start, base_seed):
    results = []
    prev_phase = None
    for f, planes in enumerate(frames):
        init_phase = prev_phase if (warm_start and prev_phase is not None) else None
        phase, history, _ = multiplane_gs_torch(
            planes, DEPTHS_M, WAVELENGTH, DX, N_ITERS_BUDGET,
            device=DEVICE, seed=base_seed * 1000 + f, pad_factor=PAD_FACTOR, init_phase=init_phase
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        results.append({"frame": f, "history": history, "phase": phase})
        prev_phase = phase
    return results


def average_speedup(frames, base_seed):
    cold = run_sequence_seeded(frames, warm_start=False, base_seed=base_seed)
    warm = run_sequence_seeded(frames, warm_start=True, base_seed=base_seed)
    rows = []
    for f in range(len(frames)):
        cold_final = cold[f]["history"][-1]
        threshold = cold_final - QUALITY_THRESHOLD_MARGIN_DB
        cold_cross = first_crossing(cold[f]["history"], threshold)
        warm_cross = first_crossing(warm[f]["history"], threshold)
        rows.append((f, cold_final, cold_cross, warm_cross))
    speedups = [c / w for _, _, c, w in rows[1:] if c and w]
    avg = sum(speedups) / len(speedups) if speedups else None
    return rows, avg


def main():
    check_patch_applied()
    print(f"Device: {DEVICE}")
    t_start = time.time()
    base = make_realistic_multiplane_target(SHAPE)
    frames = make_frame_sequence_padcrop(base, N_FRAMES, SHIFT_PX)

    results = []
    for base_seed in BASE_SEEDS:
        print(f"\n=== base_seed={base_seed} ===")
        rows, avg = average_speedup(frames, base_seed)
        for f, cold_final, cold_cross, warm_cross in rows:
            print(f"  frame {f}: cold_final={cold_final:.2f}, cold_cross={cold_cross}, warm_cross={warm_cross}")
        print(f"  average speedup (frames 1+): {avg:.2f}x" if avg else "  n/a")
        results.append({"base_seed": base_seed, "avg": avg})

    print("\n" + "=" * 70)
    print(f"{'Base seed':<12}{'Avg speedup':<14}")
    for r in results:
        avg_str = f"{r['avg']:.2f}x" if r["avg"] is not None else "n/a"
        print(f"{r['base_seed']:<12}{avg_str:<14}")
    print("=" * 70)

    avgs = [r["avg"] for r in results if r["avg"] is not None]
    if len(avgs) >= 2:
        spread = max(avgs) - min(avgs)
        spread_pct = spread / (sum(avgs) / len(avgs)) * 100
        print(f"\nSpeedup across {len(avgs)} seeds: {[round(a,2) for a in avgs]}")
        print(f"Spread: {spread:.2f}x ({spread_pct:.0f}% of mean)")
        print(f"\nOriginal single-seed figure in the document: ~6.0x")
        print(f"Range across seeds: {min(avgs):.2f}x-{max(avgs):.2f}x")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
