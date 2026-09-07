"""
Closes an open item flagged in 10.10/A.14.9: the resolution-relaxation
lever (~74% power reduction, no measured quality cost) was only tested on
procedurally-generated content, with an explicit caveat that "fine real-
world text or UI content near the relaxed resolution's own limit is the
case most likely to actually cost something." This project now has
exactly that -- a real photo with genuine fine Japanese caption text
(Translation/build_realistic_target.py) -- so this re-runs the same
methodology on it.

Same methodology as Compute/experiment_baseline_reconcile_and_resolution.py:
pad_factor=2.0 held fixed, 8 seeds, 400-iteration budget,
find_plateau_robust, real FFT counting (fft_counter_torch), GS algorithm
(multiplane_gs_torch, matching the established comparison exactly rather
than switching to SGD). Only the content and the two compared resolutions
differ: real photo/depth/caption content at 1024x1024 ("higher-res", a
stand-in for "current, full detail") vs. the same content built fresh at
512x512 (~25% of the pixel count, matching the ~28% ratio of the
established current-vs-relaxed comparison). Quality uses a per-pixel
content mask, not the sparse_content_bounds bounding box -- confirmed
broken for this scattered-content shape in the basic reconstruction test.

Run on your 3060 (expect ~15-20 min for 8 seeds x 2 resolutions):
    python3 experiment_realistic_resolution_relaxation.py
"""
import sys
import math
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\Documents\CGH\Demo")
from fft_counter_torch import counter
from retrieval_torch import multiplane_gs_torch
from metrics import psnr_intensity
from convergence import find_plateau_robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
PAD_FACTOR = 2.0
N_ITERS = 400
SEEDS = list(range(8))

CONDITIONS = {
    "higher-res (1024x1024, real content)": "_1024",
    "lower-res (512x512, real content, ~25% pixel count)": "",
}

TOPS_PER_W = 6.5
POWER_BUDGET_MW = 500


def load_targets(tag):
    return [
        np.load(f"target_near{tag}.npy"),
        np.load(f"target_mid_with_caption{tag}.npy"),
        np.load(f"target_far{tag}.npy"),
    ]


def analytic_gflop_per_fft(n_pixels):
    return 5 * n_pixels * math.log2(n_pixels) / 1e9


def masked_quality(recon_planes, targets, content_thresh=0.05):
    """Per-pixel content mask -- see test_realistic_reconstruction.py for why
    the bounding-box sparse_content_bounds mask doesn't work for this content."""
    psnrs = []
    for r, t in zip(recon_planes, targets):
        mask = t > content_thresh
        psnrs.append(psnr_intensity(r[mask], t[mask]))
    return psnrs


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()

    all_results = {}
    for cond_name, tag in CONDITIONS.items():
        print(f"\n{'='*20} {cond_name} {'='*20}")
        targets = load_targets(tag)
        shape = targets[0].shape
        gflop_per_fft = analytic_gflop_per_fft(shape[0] * PAD_FACTOR * shape[1] * PAD_FACTOR)
        print(f"Shape: {shape}, analytic GFLOP/FFT at padded size: {gflop_per_fft:.3f}")

        print("Warming up GPU...")
        _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                                 pad_factor=PAD_FACTOR, smooth_cutoff=True)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        results = []
        for seed in SEEDS:
            counter.reset()
            t0 = time.time()
            phase, history, recon_by_checkpoint = multiplane_gs_torch(
                targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, seed=seed,
                pad_factor=PAD_FACTOR, smooth_cutoff=True
            )
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            total_ffts = counter.count
            ffts_per_iter = total_ffts / N_ITERS

            plateau_iter, reference, still_rising = find_plateau_robust(history)
            ffts_at_plateau = ffts_per_iter * plateau_iter
            gflop_per_frame = ffts_at_plateau * gflop_per_fft
            tflops_needed = gflop_per_frame * 360 / 1000
            power_mw = tflops_needed / TOPS_PER_W * 1000
            budget_ratio = power_mw / POWER_BUDGET_MW

            final_checkpoint = max(recon_by_checkpoint.keys())
            corrected_psnrs = masked_quality(recon_by_checkpoint[final_checkpoint], targets)
            corrected_avg = sum(corrected_psnrs) / len(corrected_psnrs)

            results.append({
                "seed": seed, "plateau_iter": plateau_iter, "still_rising": still_rising,
                "ffts_at_plateau": ffts_at_plateau, "power_mw": power_mw,
                "budget_ratio": budget_ratio, "corrected_avg": corrected_avg,
                "corrected_psnrs": corrected_psnrs, "elapsed": elapsed,
            })
            print(f"   seed={seed}: plateau@{plateau_iter}/{N_ITERS}, corrected quality="
                  f"{corrected_avg:.2f} dB (per-plane: {[round(p,2) for p in corrected_psnrs]}), "
                  f"{ffts_at_plateau:.0f} FFTs/frame, {power_mw:.0f}mW ({budget_ratio:.1f}x budget), "
                  f"{elapsed:.1f}s{' STILL RISING' if still_rising else ''}")

        all_results[cond_name] = results

    print("\n" + "=" * 110)
    print(f"{'Condition':<50}{'FFTs/frame avg':<18}{'Power avg':<14}{'x budget':<12}{'Quality avg'}")
    summary = {}
    for cond_name, results in all_results.items():
        ffts = [r["ffts_at_plateau"] for r in results]
        powers = [r["power_mw"] for r in results]
        ratios = [r["budget_ratio"] for r in results]
        quals = [r["corrected_avg"] for r in results]
        summary[cond_name] = {"avg_ffts": np.mean(ffts), "avg_power": np.mean(powers),
                                "avg_ratio": np.mean(ratios), "avg_qual": np.mean(quals),
                                "min_qual": min(quals), "max_qual": max(quals)}
        print(f"{cond_name:<50}{np.mean(ffts):<18.0f}{np.mean(powers):<14.0f}{np.mean(ratios):<12.1f}"
              f"{np.mean(quals):.2f} ({min(quals):.2f}-{max(quals):.2f})")
    print("=" * 110)

    names = list(CONDITIONS.keys())
    compute_ratio = 100 * summary[names[1]]['avg_ffts'] / summary[names[0]]['avg_ffts']
    power_ratio = 100 * summary[names[1]]['avg_power'] / summary[names[0]]['avg_power']
    qual_delta = summary[names[1]]['avg_qual'] - summary[names[0]]['avg_qual']
    print(f"\nLower-res vs. higher-res, REAL content: {compute_ratio:.0f}% of FFTs/frame, "
          f"{power_ratio:.0f}% of power, quality delta {qual_delta:+.2f} dB")
    print(f"For comparison, procedural content (10.10/A.14.9): 103% of FFTs/frame, 26% of power, "
          f"quality delta +0.49 dB")

    any_rising = any(r["still_rising"] for results in all_results.values() for r in results)
    if any_rising:
        print("\nWARNING: at least one run was still rising at N_ITERS -- extend budget before trusting fully.")

    print("\nInterpretation: a real, safe resolution reduction needs LOWER or equal compute/power AND quality")
    print("within noise of the higher-res baseline (~0.2-0.3dB, this project's usual seed-to-seed spread).")
    if power_ratio < 50 and qual_delta > -0.3:
        print(f"CONFIRMED on real fine-text content too: ~{100-power_ratio:.0f}% less power, quality holds.")
    elif qual_delta < -0.3:
        print(f"REVERSED on real fine-text content: relaxing resolution costs real quality ({qual_delta:+.2f} dB) "
              f"-- the fine Japanese caption text IS the case that costs something, as flagged in 10.10.")
    else:
        print("Mixed or inconclusive -- report both numbers plainly rather than picking a side.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cond_name in names:
        ffts = [r["ffts_at_plateau"] for r in all_results[cond_name]]
        quals = [r["corrected_avg"] for r in all_results[cond_name]]
        axes[0].scatter([cond_name] * len(ffts), ffts, alpha=0.7)
        axes[0].scatter([cond_name], [np.mean(ffts)], color="red", marker="_", s=400, linewidths=2)
        axes[1].scatter([cond_name] * len(quals), quals, alpha=0.7)
        axes[1].scatter([cond_name], [np.mean(quals)], color="red", marker="_", s=400, linewidths=2)
    axes[0].set_ylabel("FFTs/frame at plateau")
    axes[0].tick_params(axis='x', rotation=10)
    axes[1].set_ylabel("Corrected quality (dB, per-pixel mask)")
    axes[1].tick_params(axis='x', rotation=10)
    plt.suptitle(f"Resolution relaxation on REAL content, {len(SEEDS)} seeds, {N_ITERS}-iter budget")
    plt.tight_layout()
    plt.savefig("realistic_resolution_relaxation.png", dpi=130)
    print("Saved plot: realistic_resolution_relaxation.png")


if __name__ == "__main__":
    main()
