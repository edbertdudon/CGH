"""
Follow-up to experiment_temporal_warmstart_fullres.py -- the same
multi-seed stress test that retracted the INT8 iteration-count concern
and found the 10.1c compute number unreliable, applied here.

The full-resolution warm-start speedup numbers (10.00x at ~0.6% shift,
6.00x at ~11.7% shift) were computed with first_crossing(): the first
iteration where a convergence history crosses a fixed dB threshold --
structurally the SAME kind of measurement as the plateau detector that
just proved unreliable (30-65 iteration spread across seeds on the exact
same resolution/content). first_crossing() has never been seed-stress-
tested. This checks whether it's stable or just as fragile.

Reruns the smaller (~0.6%, 28px) full-resolution shift case from
_fullres.py across 3 different base seeds (offsetting every frame's cold
-start seed by base_seed*1000, so cold-start phases differ but the
underlying content/motion is identical across runs) and reports
cold_cross/warm_cross/speedup per seed, same as the INT8 check did for
plateau_iter.

N_FRAMES reduced to 4 (from 5 in _fullres.py) to keep 3-seed runtime
reasonable at full resolution.

Run on your 3060 (expect several minutes):
    python3 experiment_temporal_warmstart_seed_check.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
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
SHIFT_PX = 28   # ~0.6% of frame width, the smaller/cleaner case from _fullres.py
BASE_SEEDS = [0, 1, 2]


def run_sequence_seeded(frames, warm_start, base_seed):
    """Same as experiment_temporal_warmstart.run_sequence, but with an
    explicit base_seed offset so cold-start phases vary across runs
    instead of always using seed=frame_index."""
    results = []
    prev_phase = None
    for f, planes in enumerate(frames):
        counter.reset()
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
        results.append({"base_seed": base_seed, "avg": avg, "rows": rows})

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
        if spread_pct > 40:
            print("\nLarge spread -- same pattern as the plateau-detector fragility found in the INT8 and")
            print("10.1c checks. The single-seed 10.00x full-res figure should NOT be treated as a stable")
            print("number; report a range/average across seeds instead.")
        else:
            print("\nSpread is more modest than the plateau-detector cases -- first_crossing() against a")
            print("threshold defined relative to cold's OWN final value (not an absolute dB target) may be")
            print("more robust than the plateau detector was. Still worth reporting as a range, not a point.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
