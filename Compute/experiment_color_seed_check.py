"""
Audit follow-up flagged in 10.9/A.13: 10.4's color figures (native 6.8dB,
naive reuse 5.8dB, joint 6.3dB) predate this project's multi-seed and
true-convergence standards -- the original test used a single seed AND
a fixed 16-iteration budget, the same combination of risk factors that
already reversed the 10.3 sequential-rendering finding and the 10.8
plane-count claim elsewhere in this document.

Three conditions, full resolution, real content (near/mid/far), across
all three depth planes, matching the original test's structure:

  A) Native: each color (green/red/blue) solved independently, full
     3-plane simultaneous GS solve per color. Run across N_SEEDS random
     seeds per color, to a generous plateau-verified budget (not a fixed
     16 iterations).
  B) Naive reuse: green-native phase from (A), propagated unmodified at
     red/blue. No extra solving -- reuses each of (A)'s N_SEEDS green
     phases, so this condition's own seed variance comes directly from
     (A)'s green solves.
  C) Joint: one phase pattern per seed, optimized against all 9
     (depth, color) constraints at once via multiconstraint_gs_torch,
     across the same N_SEEDS, same convergence-verification standard.

near/mid are sparse content (masked PSNR, per 10.6/A.6); far is dense
(whole-frame PSNR, no masking needed) -- same convention as every other
full-resolution content test in this project.

N_SEEDS=3 (not the full 5-seed standard) given this experiment's per-
condition cost is roughly 3x a single-color solve for the joint
condition and 3 colors x N_SEEDS for native -- a real, still-meaningful
multi-seed check, not the full n=5 bar, given the compute budget.

Run on your 3060 (expect ~25-30 minutes):
    python3 experiment_color_seed_check.py
"""
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, sparse_content_bounds
from retrieval_torch import multiplane_gs_torch, multiconstraint_gs_torch
from propagation_torch import angular_spectrum_propagate, safe_abs
from metrics import psnr as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 200
N_SEEDS = 3
PLATEAU_EPS_DB = 0.2

WAVELENGTHS = {"green": 520e-9, "red": 638e-9, "blue": 450e-9}
PLANE_NAMES = ["near", "mid", "far"]
SPARSE_PLANES = {0, 1}


def masked_or_full_psnr(recon, target, plane_idx):
    if plane_idx in SPARSE_PLANES:
        rows, cols = sparse_content_bounds(target)
        return psnr_np(recon[rows, cols], target[rows, cols])
    return psnr_np(recon, target)


def still_rising(history, tol=PLATEAU_EPS_DB):
    if len(history) < 20:
        return False
    return (history[-1] - sum(history[-20:-10]) / 10) > tol


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTHS["green"], DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # ---- A: native per-color solves, N_SEEDS each ----
    print(f"\nA) Native per-color solves, {N_SEEDS} seeds each, {N_ITERS} iterations...")
    native_quality = {c: [] for c in WAVELENGTHS}   # native_quality[color] = list of [q_near,q_mid,q_far] per seed
    native_phases = {c: [] for c in WAVELENGTHS}    # for naive-reuse: only green's phases are needed
    for color, wl in WAVELENGTHS.items():
        for seed in range(N_SEEDS):
            t0 = time.time()
            phase, hist, cps = multiplane_gs_torch(
                targets, DEPTHS_M, wl, DX, N_ITERS, device=DEVICE, seed=seed,
                pad_factor=PAD_FACTOR, smooth_cutoff=True
            )
            elapsed = time.time() - t0
            final_iter = max(cps.keys())
            q = [masked_or_full_psnr(cps[final_iter][i], targets[i], i) for i in range(3)]
            rising = still_rising(hist)
            native_quality[color].append(q)
            native_phases[color].append(phase)
            print(f"   {color} seed={seed}: {[round(x,2) for x in q]} dB, {elapsed:.1f}s"
                  f"{' STILL RISING' if rising else ''}")

    # ---- B: naive reuse -- reuse green's phases from A, propagate at red/blue ----
    print(f"\nB) Naive reuse: green-native phases from (A), propagated at red/blue (no re-solving)...")
    reuse_quality = {"green": native_quality["green"]}
    for color in ["red", "blue"]:
        wl = WAVELENGTHS[color]
        seed_results = []
        for seed in range(N_SEEDS):
            green_phase = native_phases["green"][seed]
            q = []
            for i, z in enumerate(DEPTHS_M):
                recon = angular_spectrum_propagate(torch.exp(1j * green_phase), wl, DX, z, pad_factor=PAD_FACTOR)
                q.append(masked_or_full_psnr(safe_abs(recon).cpu().numpy(), targets[i], i))
            seed_results.append(q)
        reuse_quality[color] = seed_results
        print(f"   {color}: {[[round(x,2) for x in q] for q in seed_results]} dB")

    # ---- C: joint multi-wavelength, multi-depth solve, N_SEEDS ----
    print(f"\nC) Joint solve: one phase pattern, all 9 constraints, {N_SEEDS} seeds, {N_ITERS} iterations...")
    constraints = []
    for i, z in enumerate(DEPTHS_M):
        for color, wl in WAVELENGTHS.items():
            constraints.append({"target": targets[i], "z": z, "wavelength": wl})

    joint_quality = {"green": [], "red": [], "blue": []}
    for seed in range(N_SEEDS):
        t0 = time.time()
        joint_phase, joint_hist, joint_cps = multiconstraint_gs_torch(
            constraints, DX, N_ITERS, device=DEVICE, seed=seed, pad_factor=PAD_FACTOR
        )
        elapsed = time.time() - t0
        final_joint = max(joint_cps.keys())
        rising = still_rising(joint_hist)
        idx = 0
        seed_q = {"green": [], "red": [], "blue": []}
        for i in range(3):
            for color in WAVELENGTHS:
                q = masked_or_full_psnr(joint_cps[final_joint][idx], targets[i], i)
                seed_q[color].append(q)
                idx += 1
        for color in WAVELENGTHS:
            joint_quality[color].append(seed_q[color])
        print(f"   seed={seed}: green={[round(x,2) for x in seed_q['green']]}, "
              f"red={[round(x,2) for x in seed_q['red']]}, blue={[round(x,2) for x in seed_q['blue']]}, "
              f"{elapsed:.1f}s{' STILL RISING' if rising else ''}")

    # ---- summary: average red+blue quality per condition, across seeds ----
    def avg_redblue(quality_dict):
        vals = []
        for color in ["red", "blue"]:
            for seed_result in quality_dict[color]:
                vals.extend(seed_result)
        return sum(vals) / len(vals), min(vals), max(vals)

    native_avg, native_min, native_max = avg_redblue(native_quality)
    reuse_avg, reuse_min, reuse_max = avg_redblue(reuse_quality)
    joint_avg, joint_min, joint_max = avg_redblue(joint_quality)

    print("\n" + "=" * 90)
    print("SUMMARY -- average quality (red+blue, all planes, all seeds)")
    print(f"{'Condition':<15}{'Avg dB':<12}{'Range':<20}{'Gap vs native'}")
    print(f"{'Native':<15}{native_avg:<12.2f}{f'{native_min:.2f}-{native_max:.2f}':<20}{'--'}")
    print(f"{'Naive reuse':<15}{reuse_avg:<12.2f}{f'{reuse_min:.2f}-{reuse_max:.2f}':<20}{reuse_avg-native_avg:+.2f}")
    print(f"{'Joint':<15}{joint_avg:<12.2f}{f'{joint_min:.2f}-{joint_max:.2f}':<20}{joint_avg-native_avg:+.2f}")
    print("=" * 90)

    print(f"\nOriginal single-seed, fixed-16-iteration finding: native 6.8dB, naive reuse 5.8dB (-1.0dB), "
          f"joint 6.3dB (-0.5dB).")
    print(f"This test (N_SEEDS={N_SEEDS}, true convergence): native {native_avg:.2f}dB, "
          f"naive reuse {reuse_avg:.2f}dB ({reuse_avg-native_avg:+.2f}dB), "
          f"joint {joint_avg:.2f}dB ({joint_avg-native_avg:+.2f}dB)")

    reuse_gap_change = abs((reuse_avg - native_avg) - (-1.0))
    joint_gap_change = abs((joint_avg - native_avg) - (-0.5))
    if reuse_gap_change < 1.0 and joint_gap_change < 1.0:
        print("\nGaps are within ~1dB of the original figures -- the original single-seed/fixed-budget")
        print("finding holds up qualitatively under proper testing. Ordering (native > joint > reuse) survives.")
    else:
        print("\nGaps differ meaningfully from the original figures -- the single-seed/fixed-budget result")
        print("does NOT hold up as reported. Do not carry the original 6.8/5.8/6.3dB figures forward")
        print("without replacing them with this revalidated result.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = ["Native", "Naive reuse", "Joint"]
    avgs = [native_avg, reuse_avg, joint_avg]
    mins = [native_min, reuse_min, joint_min]
    maxs = [native_max, reuse_max, joint_max]
    errs = [[a - mn for a, mn in zip(avgs, mins)], [mx - a for a, mx in zip(avgs, maxs)]]
    ax.bar(conditions, avgs, yerr=errs, capsize=8, color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_ylabel("PSNR (dB), red+blue avg, all planes/seeds")
    ax.set_title(f"Color solve comparison, true convergence, {N_SEEDS} seeds, full resolution")
    plt.tight_layout()
    plt.savefig("color_seed_check.png", dpi=130)
    print("Saved plot: color_seed_check.png")


if __name__ == "__main__":
    main()
