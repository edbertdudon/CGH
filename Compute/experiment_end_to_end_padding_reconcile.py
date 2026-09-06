"""
Reconciles two things before the padding-factor result (see
experiment_padding_factor_fullres.py) can honestly update the document's
headline 10.1 compute baseline:

1. That baseline (552 +/- 198 FFTs/frame, 5 seeds) was measured by
   experiment_end_to_end_seed_check.py using the ORIGINAL first-touch
   plateau detector (find_plateau: first iteration >= final-tol) capped
   at N_ITERS=100.
2. experiment_padding_factor_fullres.py used the newer sustained-hold
   detector (find_plateau_robust, Demo/convergence.py) over a 300-iteration
   budget -- and its own pad_factor=2.0 condition came back at ~1484
   FFTs/frame average, nearly 3x the official 552 baseline, for what
   should be the identical setup.

Those numbers are not directly comparable: a different detector AND a
3x longer budget both push detected plateau later, so the padding
experiment's internal ratio (pad=1.3 costs ~47% of its OWN pad=2.0
reading) cannot simply be multiplied onto the official 552-FFT figure
without reconciling the methodology gap first.

This measures pad_factor=2.0 and pad_factor=1.3 side by side, 5 seeds
each (matching the official baseline's seed count), same detector
(find_plateau_robust) and same generous budget (300 iterations) for
both -- so whatever the padding lever's real effect turns out to be, it
is measured on a clean, internally-consistent basis, and any shift in
the pad=2.0 number itself (from the detector/budget change alone) is
visible and reported honestly rather than folded silently into the
padding conclusion.

Run on your 3060 (expect ~35-45 min: 2 pad_factors x 5 seeds, full res):
    python3 experiment_end_to_end_padding_reconcile.py
"""
import math
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch
from convergence import find_plateau_robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 300
SEEDS = [0, 1, 2, 3, 4]
PAD_FACTORS = [2.0, 1.3]

TOPS_PER_W = 6.5
POWER_BUDGET_MW = 500


def analytic_gflop_per_fft(n_pixels):
    return 5 * n_pixels * math.log2(n_pixels) / 1e9


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=2.0, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    all_results = {}
    for pf in PAD_FACTORS:
        print(f"\n{'='*20} pad_factor={pf} {'='*20}")
        gflop_per_fft = analytic_gflop_per_fft(SHAPE[0] * pf * SHAPE[1] * pf)
        print(f"Analytic GFLOP/FFT at this padded size: {gflop_per_fft:.3f}")

        results = []
        for seed in SEEDS:
            counter.reset()
            t0 = time.time()
            phase, history, cps = multiplane_gs_torch(
                targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, seed=seed,
                pad_factor=pf, smooth_cutoff=True
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

            results.append({
                "seed": seed, "plateau_iter": plateau_iter, "final_psnr": history[-1],
                "still_rising": still_rising, "ffts_at_plateau": ffts_at_plateau,
                "power_mw": power_mw, "budget_ratio": budget_ratio, "elapsed": elapsed,
            })
            print(f"   seed={seed}: plateau@{plateau_iter}/{N_ITERS}, final {history[-1]:.2f} dB, "
                  f"{ffts_at_plateau:.0f} FFTs/frame, {power_mw:.0f}mW ({budget_ratio:.1f}x budget), "
                  f"{elapsed:.1f}s{' STILL RISING -- N_ITERS may be too low' if still_rising else ''}")

        all_results[pf] = results

    print("\n" + "=" * 100)
    print(f"{'pad_factor':<12}{'FFTs/frame (avg)':<20}{'Range':<20}{'Power avg (mW)':<18}{'x budget avg'}")
    summary = {}
    for pf in PAD_FACTORS:
        results = all_results[pf]
        ffts = [r["ffts_at_plateau"] for r in results]
        powers = [r["power_mw"] for r in results]
        ratios = [r["budget_ratio"] for r in results]
        avg_ffts = sum(ffts) / len(ffts)
        avg_power = sum(powers) / len(powers)
        avg_ratio = sum(ratios) / len(ratios)
        summary[pf] = {"avg_ffts": avg_ffts, "avg_power": avg_power, "avg_ratio": avg_ratio,
                        "min_ffts": min(ffts), "max_ffts": max(ffts)}
        print(f"{pf:<12}{avg_ffts:<20.0f}{f'{min(ffts):.0f}-{max(ffts):.0f}':<20}{avg_power:<18.0f}{avg_ratio:.1f}x")
    print("=" * 100)

    print(f"\nOfficial document baseline (pad=2.0, first-touch detector, 100-iter budget): 552 FFTs/frame, ~199W, ~397x")
    print(f"This measurement (pad=2.0, sustained-hold detector, 300-iter budget): "
          f"{summary[2.0]['avg_ffts']:.0f} FFTs/frame, {summary[2.0]['avg_power']:.0f}mW, {summary[2.0]['avg_ratio']:.1f}x")
    delta_pct = 100 * (summary[2.0]['avg_ffts'] / 552 - 1)
    print(f"Detector/budget-methodology delta at pad=2.0: {delta_pct:+.0f}% vs. the official baseline")

    print(f"\npad=1.3 (this measurement): {summary[1.3]['avg_ffts']:.0f} FFTs/frame, "
          f"{summary[1.3]['avg_power']:.0f}mW, {summary[1.3]['avg_ratio']:.1f}x")
    rel_to_own_pad2 = 100 * summary[1.3]['avg_ffts'] / summary[2.0]['avg_ffts']
    print(f"pad=1.3 vs. THIS measurement's own pad=2.0: {rel_to_own_pad2:.0f}% of FFTs/frame "
          f"(clean, same-methodology comparison)")

    projected_ffts = 552 * (summary[1.3]['avg_ffts'] / summary[2.0]['avg_ffts'])
    projected_power = 199 * (summary[1.3]['avg_ffts'] / summary[2.0]['avg_ffts'])
    projected_ratio = 397 * (summary[1.3]['avg_ffts'] / summary[2.0]['avg_ffts'])
    print(f"\nIf applying pad=1.3's clean relative ratio ({rel_to_own_pad2:.0f}%) onto the OFFICIAL")
    print(f"552-FFT baseline (not this run's own inflated pad=2.0 number): "
          f"~{projected_ffts:.0f} FFTs/frame, ~{projected_power:.0f}mW, ~{projected_ratio:.0f}x budget")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(7, 5))
    for pf in PAD_FACTORS:
        vals = [r["ffts_at_plateau"] for r in all_results[pf]]
        ax.scatter([str(pf)] * len(vals), vals, alpha=0.7)
        ax.scatter([str(pf)], [sum(vals) / len(vals)], color="red", marker="_", s=400, linewidths=2)
    ax.axhline(552, color="gray", linestyle=":", label="official baseline (552, old detector/budget)")
    ax.set_ylabel("FFTs/frame at plateau")
    ax.set_xlabel("pad_factor")
    ax.set_title("Padding reconciliation: pad=2.0 vs 1.3, robust detector, 300-iter budget, 5 seeds")
    ax.legend()
    plt.tight_layout()
    plt.savefig("end_to_end_padding_reconcile.png", dpi=130)
    print("Saved plot: end_to_end_padding_reconcile.png")


if __name__ == "__main__":
    main()
