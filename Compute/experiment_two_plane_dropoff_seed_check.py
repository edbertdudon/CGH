"""
Seed-check follow-up to experiment_two_plane_dropoff_fullres.py, before
trusting its compute-side result (2-plane needing ~3x more iterations
than 3-plane, flipping a claimed +22.7% compute saving into a ~97%
compute cost). That comparison was single-seed on both sides -- exactly
the kind of measurement this project's whole audit exists to distrust.
This reruns both conditions across 3 seeds each to see whether the
64-vs-189-iteration gap is a real, systematic property of dropping mid,
or partly/mostly ordinary seed variance (30-65+ iteration swings are
already established as normal for full-resolution GS convergence in
this project).

Run on your 3060 (expect ~15 minutes):
    python3 experiment_two_plane_dropoff_seed_check.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_BUDGET = 250
PLATEAU_EPS_DB = 0.1
SEEDS = [0, 1, 2]


def find_plateau_robust(history, eps=PLATEAU_EPS_DB):
    final = history[-1]
    for i in range(len(history)):
        if all(abs(v - final) < eps for v in history[i:]):
            return i + 1
    return len(history)


def solve(targets_subset, depths_subset, seed):
    t0 = time.time()
    phase, history, _ = multiplane_gs_torch(
        targets_subset, depths_subset, WAVELENGTH, DX, N_ITERS_BUDGET,
        device=DEVICE, seed=seed, pad_factor=PAD_FACTOR, smooth_cutoff=True
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    plateau = find_plateau_robust(history)
    still_rising = len(history) >= 20 and (history[-1] - sum(history[-20:-10]) / 10) > PLATEAU_EPS_DB
    ffts_per_iter = 4 * len(depths_subset)
    return {"plateau": plateau, "final": history[-1], "ffts": plateau * ffts_per_iter,
            "elapsed": elapsed, "still_rising": still_rising}


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)
    near, mid, far = targets

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print(f"\n3-plane baseline, {len(SEEDS)} seeds...")
    baseline_results = []
    for seed in SEEDS:
        r = solve(targets, DEPTHS_M, seed)
        baseline_results.append(r)
        print(f"  seed={seed}: plateau={r['plateau']}, FFTs={r['ffts']}, final={r['final']:.2f} dB, "
              f"{r['elapsed']:.1f}s{' STILL RISING' if r['still_rising'] else ''}")

    print(f"\n2-plane (mid dropped), {len(SEEDS)} seeds...")
    two_plane_results = []
    for seed in SEEDS:
        r = solve([near, far], [DEPTHS_M[0], DEPTHS_M[2]], seed)
        two_plane_results.append(r)
        print(f"  seed={seed}: plateau={r['plateau']}, FFTs={r['ffts']}, final={r['final']:.2f} dB, "
              f"{r['elapsed']:.1f}s{' STILL RISING' if r['still_rising'] else ''}")

    baseline_ffts = [r["ffts"] for r in baseline_results]
    two_plane_ffts = [r["ffts"] for r in two_plane_results]
    baseline_avg = sum(baseline_ffts) / len(baseline_ffts)
    two_plane_avg = sum(two_plane_ffts) / len(two_plane_ffts)

    print("\n" + "=" * 80)
    print(f"{'Condition':<20}{'FFTs per seed':<30}{'Avg FFTs':<12}{'Range'}")
    print(f"{'3-plane':<20}{str(baseline_ffts):<30}{baseline_avg:<12.0f}{min(baseline_ffts)}-{max(baseline_ffts)}")
    print(f"{'2-plane':<20}{str(two_plane_ffts):<30}{two_plane_avg:<12.0f}{min(two_plane_ffts)}-{max(two_plane_ffts)}")
    print("=" * 80)

    ffts_saved_pct_avg = 100 * (1 - two_plane_avg / baseline_avg)
    print(f"\nSingle-seed result being checked: 3-plane=768 FFTs, 2-plane=1512 FFTs (-96.9%, i.e. ~2x cost increase)")
    print(f"3-seed average: 3-plane={baseline_avg:.0f} FFTs, 2-plane={two_plane_avg:.0f} FFTs "
          f"({ffts_saved_pct_avg:+.1f}%)")

    # does the range of plausible outcomes (mixing any baseline seed with any two-plane seed)
    # ever produce a real saving, or is 2-plane consistently more expensive regardless of pairing?
    all_pairs_pct = [100 * (1 - tp / bl) for bl in baseline_ffts for tp in two_plane_ffts]
    print(f"\nAcross all {len(all_pairs_pct)} seed pairings: {[round(p,1) for p in all_pairs_pct]}% "
          f"(range {min(all_pairs_pct):+.1f}% to {max(all_pairs_pct):+.1f}%)")

    if all(p < 0 for p in all_pairs_pct):
        print("\n2-plane costs MORE than 3-plane in every single seed pairing tested -- the compute-side")
        print("reversal at full resolution is real and consistent, not a single-seed artifact. The original")
        print("512x512 finding (+22.7% savings) does not hold at target resolution; report the full-res")
        print("figure as a genuine cost increase, not a saving.")
    elif all(p > 0 for p in all_pairs_pct):
        print("\n2-plane saves compute in every seed pairing -- the single-seed full-res result (-96.9%) was")
        print("itself the outlier/artifact; the original small-scale finding (savings) actually holds once")
        print("seed variance is accounted for.")
    else:
        print("\nMixed: sign depends on which seeds get paired -- neither 'saves compute' nor 'costs more'")
        print("is reliable at full resolution without averaging across seeds; report a range, not a point,")
        print("and flag this as genuinely undetermined by a single-seed comparison.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
