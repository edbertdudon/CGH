"""
Two questions combined into one properly-powered measurement:

1) The 10.1 baseline (552 FFTs/frame, 5 seeds) was measured with the
   older first-touch plateau detector capped at 100 iterations. A
   reconciliation run (experiment_end_to_end_padding_reconcile.py, 5
   seeds, robust detector, 300-iter budget) found the CURRENT-resolution
   pad=2.0 condition averaging 1870 FFTs/frame -- 3.4x higher -- with
   individual seeds ranging 769-2644, too noisy to trust from only 5
   seeds. This reruns the current resolution with more seeds (8) and a
   longer budget (400 iterations, since some histories were still
   drifting at iteration 220-300) to get a trustworthy corrected
   baseline.

2) Comparing our target resolution against the peer-reviewed synthetic-
   aperture waveguide holography paper (Reference/Choi_2024...pdf):
   our derived target is ~0.63 arcmin/pixel, nearly 2x finer than both
   that paper's demonstrated 1.2 arcmin resolution and the ~1 arcmin
   limit of human 20/20 vision. Relaxing our resolution target to match
   1.2 arcmin/pixel (real-world-validated, not just a theoretical
   perceptual limit) cuts total pixel count to ~28% of the current
   target (2,475x1,422 vs 4,700x2,700). If quality holds up on real
   content, this is a compute lever independent of and probably larger
   than padding or INT8 -- but "if" needs checking on the same content
   this project uses everywhere else, not assumed for free.

Same methodology for both conditions (pad_factor=2.0 held fixed, so the
resolution comparison isn't confounded by the still-unresolved padding
question): 8 seeds, 400-iteration budget, find_plateau_robust, real FFT
counting (fft_counter_torch, not an assumed FFTs-per-iteration formula),
and the corrected + masked PSNR metric (sparse_content_bounds for
near/mid, whole-frame for far) established during the plane-count audit.

Run on your 3060 (expect ~35-45 min):
    python3 experiment_baseline_reconcile_and_resolution.py
"""
import math
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target, sparse_content_bounds
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
    "current (2,700x4,700, ~0.63 arcmin/px)": (2700, 4700),
    "relaxed (1,422x2,475, ~1.2 arcmin/px)": (1422, 2475),
}

TOPS_PER_W = 6.5
POWER_BUDGET_MW = 500


def analytic_gflop_per_fft(n_pixels):
    return 5 * n_pixels * math.log2(n_pixels) / 1e9


def masked_quality(recon_planes, targets):
    psnrs = []
    for i, (r, t) in enumerate(zip(recon_planes, targets)):
        if i in (0, 1):  # near, mid -- sparse, mask to content bounds
            rows, cols = sparse_content_bounds(t)
            psnrs.append(psnr_intensity(r[rows, cols], t[rows, cols]))
        else:
            psnrs.append(psnr_intensity(r, t))
    return psnrs


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()

    all_results = {}
    for cond_name, shape in CONDITIONS.items():
        print(f"\n{'='*20} {cond_name} {'='*20}")
        targets = make_realistic_multiplane_target(shape)
        gflop_per_fft = analytic_gflop_per_fft(shape[0] * PAD_FACTOR * shape[1] * PAD_FACTOR)
        print(f"Analytic GFLOP/FFT at this padded size: {gflop_per_fft:.3f}")

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
                  f"{elapsed:.1f}s{' STILL RISING -- N_ITERS may be too low' if still_rising else ''}")

        all_results[cond_name] = results

    print("\n" + "=" * 110)
    print(f"{'Condition':<42}{'FFTs/frame avg':<18}{'Range':<18}{'Power avg':<14}{'x budget':<12}{'Quality avg'}")
    summary = {}
    for cond_name, results in all_results.items():
        ffts = [r["ffts_at_plateau"] for r in results]
        powers = [r["power_mw"] for r in results]
        ratios = [r["budget_ratio"] for r in results]
        quals = [r["corrected_avg"] for r in results]
        avg_ffts = sum(ffts) / len(ffts)
        avg_power = sum(powers) / len(powers)
        avg_ratio = sum(ratios) / len(ratios)
        avg_qual = sum(quals) / len(quals)
        summary[cond_name] = {"avg_ffts": avg_ffts, "avg_power": avg_power,
                                "avg_ratio": avg_ratio, "avg_qual": avg_qual,
                                "min_ffts": min(ffts), "max_ffts": max(ffts),
                                "min_qual": min(quals), "max_qual": max(quals)}
        print(f"{cond_name:<42}{avg_ffts:<18.0f}{f'{min(ffts):.0f}-{max(ffts):.0f}':<18}"
              f"{avg_power:<14.0f}{avg_ratio:<12.1f}{avg_qual:.2f} ({min(quals):.2f}-{max(quals):.2f})")
    print("=" * 110)

    names = list(CONDITIONS.keys())
    print(f"\nOfficial document baseline (old detector/100-iter budget, 5 seeds): 552 FFTs/frame, ~199W, ~397x")
    print(f"This measurement, current resolution (robust detector/{N_ITERS}-iter budget, {len(SEEDS)} seeds): "
          f"{summary[names[0]]['avg_ffts']:.0f} FFTs/frame, {summary[names[0]]['avg_power']:.0f}mW, "
          f"{summary[names[0]]['avg_ratio']:.1f}x")
    delta_pct = 100 * (summary[names[0]]['avg_ffts'] / 552 - 1)
    print(f"Delta vs. official baseline: {delta_pct:+.0f}%")

    print(f"\nRelaxed resolution: {summary[names[1]]['avg_ffts']:.0f} FFTs/frame, "
          f"{summary[names[1]]['avg_power']:.0f}mW, {summary[names[1]]['avg_ratio']:.1f}x")
    compute_ratio = 100 * summary[names[1]]['avg_ffts'] / summary[names[0]]['avg_ffts']
    power_ratio = 100 * summary[names[1]]['avg_power'] / summary[names[0]]['avg_power']
    qual_delta = summary[names[1]]['avg_qual'] - summary[names[0]]['avg_qual']
    print(f"Relaxed vs. current (this measurement, clean same-methodology comparison): "
          f"{compute_ratio:.0f}% of FFTs/frame, {power_ratio:.0f}% of power, quality delta {qual_delta:+.2f} dB")

    any_rising = any(r["still_rising"] for results in all_results.values() for r in results)
    if any_rising:
        print("\nWARNING: at least one run was still rising at N_ITERS -- consider a longer budget before")
        print("fully trusting the averages above.")

    print("\nInterpretation: a real, safe resolution reduction needs LOWER or equal compute/power AND quality")
    print("within noise of the current baseline (~0.2-0.3dB, matching this project's usual seed-to-seed spread).")
    if compute_ratio < 50 and abs(qual_delta) < 0.3:
        print(f"\nRelaxed resolution is a genuine win: ~{100-compute_ratio:.0f}% less compute, quality within noise.")
        print("Worth confirming this holds with a dedicated content-quality check (not just this project's")
        print("procedurally-generated icon/text/photo set) before adopting as a new target resolution.")
    elif qual_delta < -0.3:
        print(f"\nRelaxed resolution costs real quality ({qual_delta:+.2f} dB) -- not a free win; the finer")
        print("target resolution may be earning its keep after all, at least for this content.")
    else:
        print("\nMixed or inconclusive -- report both numbers plainly rather than picking a side.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cond_name in names:
        ffts = [r["ffts_at_plateau"] for r in all_results[cond_name]]
        quals = [r["corrected_avg"] for r in all_results[cond_name]]
        axes[0].scatter([cond_name] * len(ffts), ffts, alpha=0.7)
        axes[0].scatter([cond_name], [sum(ffts) / len(ffts)], color="red", marker="_", s=400, linewidths=2)
        axes[1].scatter([cond_name] * len(quals), quals, alpha=0.7)
        axes[1].scatter([cond_name], [sum(quals) / len(quals)], color="red", marker="_", s=400, linewidths=2)
    axes[0].axhline(552, color="gray", linestyle=":", label="official baseline (552)")
    axes[0].set_ylabel("FFTs/frame at plateau")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=15)
    axes[1].set_ylabel("Corrected quality (dB)")
    axes[1].tick_params(axis='x', rotation=15)
    plt.suptitle(f"Baseline reconciliation + resolution relaxation, {len(SEEDS)} seeds, {N_ITERS}-iter budget")
    plt.tight_layout()
    plt.savefig("baseline_reconcile_and_resolution.png", dpi=130)
    print("Saved plot: baseline_reconcile_and_resolution.png")


if __name__ == "__main__":
    main()
