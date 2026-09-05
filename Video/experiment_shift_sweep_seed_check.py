"""
Audit follow-up (10.9): the shift-magnitude sweep's non-monotonic
speedup curve (roughly 15x at small shift, dipping to ~6x mid-sweep,
partially recovering at large shift) has been cited in this document as
"unexplained" since it was first measured -- but every point in that
sweep was a SINGLE seed, never itself re-checked against the multi-seed
standard this project now applies everywhere else. Given that standard
has already explained away several other "weird" single-seed patterns
as ordinary seed variance (10.1's compute baseline, A.7), the leading
open question here is: is the non-monotonic shape a real property of
shift magnitude, or does it disappear once each point is averaged over
multiple seeds the way every other convergence-speed claim in this
document now is?

Retests the full 6-point shift sweep (3, 10, 30, 60, 100, 150px on a
512x512 frame) with 3 base seeds per point, using pad-crop (non-
wrapping) motion -- the methodology already established as more
defensible than the original roll-based (wraparound) version this sweep
was first measured with. Small scale first, matching this project's
usual pattern; only worth a full-resolution version if a real,
seed-stable pattern survives here.

Run on your 3060 (expect a few minutes):
    python3 experiment_shift_sweep_seed_check.py
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
SHAPE = (512, 512)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_FRAMES = 8
N_ITERS_BUDGET = 30
QUALITY_THRESHOLD_MARGIN_DB = 0.5
SHIFT_VALUES_PX = [3, 10, 30, 60, 100, 150]
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
    speedups = []
    for f in range(len(frames)):
        cold_final = cold[f]["history"][-1]
        threshold = cold_final - QUALITY_THRESHOLD_MARGIN_DB
        cold_cross = first_crossing(cold[f]["history"], threshold)
        warm_cross = first_crossing(warm[f]["history"], threshold)
        if f > 0 and cold_cross and warm_cross:
            speedups.append(cold_cross / warm_cross)
    return sum(speedups) / len(speedups) if speedups else None


def main():
    check_patch_applied()
    print(f"Device: {DEVICE}")
    t_start = time.time()
    base = make_realistic_multiplane_target(SHAPE)

    results = {}
    for shift in SHIFT_VALUES_PX:
        frames = make_frame_sequence_padcrop(base, N_FRAMES, shift)
        seed_avgs = []
        for base_seed in BASE_SEEDS:
            avg = average_speedup(frames, base_seed)
            seed_avgs.append(avg)
        results[shift] = seed_avgs
        valid = [a for a in seed_avgs if a is not None]
        mean_avg = sum(valid) / len(valid) if valid else None
        spread = (max(valid) - min(valid)) if len(valid) >= 2 else 0
        print(f"shift={shift}px: seeds={[round(a,2) if a else None for a in seed_avgs]}, "
              f"mean={mean_avg:.2f}x, spread={spread:.2f}x")

    print("\n" + "=" * 80)
    print(f"{'Shift (px)':<14}{'Seed avgs':<36}{'Mean':<10}{'Within-point spread'}")
    means = []
    for shift in SHIFT_VALUES_PX:
        seed_avgs = results[shift]
        valid = [a for a in seed_avgs if a is not None]
        mean_avg = sum(valid) / len(valid) if valid else float("nan")
        spread = (max(valid) - min(valid)) if len(valid) >= 2 else 0
        means.append(mean_avg)
        print(f"{shift:<14}{str([round(a,2) if a else None for a in seed_avgs]):<36}"
              f"{mean_avg:<10.2f}{spread:.2f}")
    print("=" * 80)

    print(f"\nOriginal single-seed sweep (roll-based, superseded by pad-crop elsewhere): "
          f"14.86x, 12.07x, 12.52x, 8.31x, 9.19x, 9.05x (non-monotonic, cited as 'unexplained')")
    print(f"This test (pad-crop, 3-seed mean): {[round(m,2) for m in means]}")

    is_monotonic = all(means[i] >= means[i + 1] - 0.5 for i in range(len(means) - 1))
    max_within_point_spread = max(
        (max(v) - min(v)) for v in results.values() if len([a for a in v if a is not None]) >= 2
    )
    between_point_range = max(means) - min(means)

    print(f"\nMax within-point (seed-to-seed) spread at any single shift value: {max_within_point_spread:.2f}x")
    print(f"Between-point (across shift values) range of means: {between_point_range:.2f}x")

    if max_within_point_spread >= between_point_range * 0.7:
        print("\nWithin-point seed variance is comparable to the between-point pattern -- the non-monotonic")
        print("shape is likely mostly seed noise, not a real property of shift magnitude. The 'unexplained")
        print("non-monotonic pattern' flagged in 10.9 should be downgraded to 'not distinguishable from")
        print("seed noise at this sample size,' not treated as a real, unexplained physical effect.")
    elif is_monotonic:
        print("\nThe pattern is now monotonically decaying once seed-averaged -- the original non-monotonic")
        print("shape was itself a seed-noise artifact of the single-seed measurement, not a real effect.")
    else:
        print("\nThe non-monotonic shape survives seed-averaging and exceeds within-point seed noise --")
        print("this appears to be a real effect of shift magnitude, not just noise. Worth investigating")
        print("further (e.g. full-resolution retest) rather than dismissing.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    for base_seed_idx, base_seed in enumerate(BASE_SEEDS):
        vals = [results[shift][base_seed_idx] for shift in SHIFT_VALUES_PX]
        ax.plot(SHIFT_VALUES_PX, vals, marker="o", alpha=0.5, label=f"seed={base_seed}")
    ax.plot(SHIFT_VALUES_PX, means, marker="s", linewidth=2.5, color="black", label="mean")
    ax.axhline(1.0, color="gray", linestyle="--")
    ax.set_xlabel("Shift per frame (px, 512x512 canvas)")
    ax.set_ylabel("Avg. iteration speedup (warm vs. cold)")
    ax.set_title("Shift-sweep seed check: is the non-monotonic shape real or noise?")
    ax.legend()
    plt.tight_layout()
    plt.savefig("shift_sweep_seed_check.png", dpi=130)
    print("Saved plot: shift_sweep_seed_check.png")


if __name__ == "__main__":
    main()
