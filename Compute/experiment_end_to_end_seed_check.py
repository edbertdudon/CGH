"""
Follow-up to experiment_end_to_end_compute.py -- the multi-seed stress
test the INT8 detour just showed is necessary before trusting a
single-run plateau number.

The INT8 reproducibility check (experiment_int8_plateau_reproducibility.py)
found float32's own plateau_iter swinging from 37 to 65 across three
seeds at this exact resolution/content/algorithm -- a spread (28
iterations) bigger than the gap that originally looked like a real INT8
effect. That means the CURRENT authoritative 10.1c number (434 FFTs/
frame, 36-iteration plateau, 156W, 312.4x over budget, from
experiment_end_to_end_compute.py) is sitting on the exact same
single-seed, unstressed ground the INT8 result was two rounds ago --
confirmed deterministic (same seed reruns identically), never confirmed
STABLE across seeds.

Reruns the unquantized, full-resolution, real-content, smooth_cutoff
GS convergence sweep (identical settings to experiment_end_to_end_compute.py)
across several seeds and reports the same plateau_iter / FFTs / power
numbers for each, plus the spread -- exactly the treatment that
retracted the INT8 iteration-count concern, applied to the number
actually being proposed for Gap 2.

Run on your 3060 (expect several minutes):
    python3 experiment_end_to_end_seed_check.py
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 100
PLATEAU_TOLERANCE_DB = 0.2
SEEDS = [0, 1, 2, 3, 4]

TOPS_PER_W = 6.5
POWER_BUDGET_MW = 500


def analytic_gflop_per_fft(n_pixels):
    return 5 * n_pixels * math.log2(n_pixels) / 1e9


def find_plateau(history, tol=PLATEAU_TOLERANCE_DB):
    final = history[-1]
    plateau_iter = next(i + 1 for i, v in enumerate(history) if v >= final - tol)
    still_rising = len(history) >= 10 and (final - history[-10]) > tol
    return plateau_iter, final, still_rising


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    gflop_per_fft = analytic_gflop_per_fft(SHAPE[0] * PAD_FACTOR * SHAPE[1] * PAD_FACTOR)

    results = []
    for seed in SEEDS:
        print(f"Running seed={seed}, {N_ITERS} iterations...")
        counter.reset()
        t0 = time.time()
        phase, history, cps = multiplane_gs_torch(
            targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        total_ffts = counter.count
        ffts_per_iter = total_ffts / N_ITERS

        plateau_iter, final_psnr, still_rising = find_plateau(history)
        ffts_at_plateau = ffts_per_iter * plateau_iter
        gflop_per_frame = ffts_at_plateau * gflop_per_fft
        tflops_needed = gflop_per_frame * 360 / 1000
        power_mw = tflops_needed / TOPS_PER_W * 1000
        budget_ratio = power_mw / POWER_BUDGET_MW

        results.append({
            "seed": seed, "plateau_iter": plateau_iter, "final_psnr": final_psnr,
            "still_rising": still_rising, "ffts_at_plateau": ffts_at_plateau,
            "power_mw": power_mw, "budget_ratio": budget_ratio, "elapsed": elapsed,
            "history": history,
        })
        print(f"   plateau@{plateau_iter}/{N_ITERS}, final {final_psnr:.2f} dB, "
              f"{ffts_at_plateau:.0f} FFTs/frame, {power_mw:.0f}mW ({budget_ratio:.1f}x budget), "
              f"{elapsed:.1f}s{' STILL RISING' if still_rising else ''}")

    print("\n" + "=" * 100)
    print(f"{'Seed':<8}{'Plateau iter':<16}{'Final dB':<12}{'FFTs/frame':<14}{'Power (mW)':<14}{'x budget':<10}")
    for r in results:
        print(f"{r['seed']:<8}{r['plateau_iter']:<16}{r['final_psnr']:<12.2f}{r['ffts_at_plateau']:<14.0f}"
              f"{r['power_mw']:<14.0f}{r['budget_ratio']:<10.1f}")
    print("=" * 100)

    plateaus = [r["plateau_iter"] for r in results]
    finals = [r["final_psnr"] for r in results]
    powers = [r["power_mw"] for r in results]
    plateau_spread = max(plateaus) - min(plateaus)
    final_spread = max(finals) - min(finals)
    power_spread_pct = (max(powers) - min(powers)) / (sum(powers) / len(powers)) * 100

    print(f"\nPlateau iteration across {len(SEEDS)} seeds: {plateaus} (spread: {plateau_spread} iterations)")
    print(f"Final quality spread: {final_spread:.2f} dB")
    print(f"Implied power spread: {min(powers):.0f}-{max(powers):.0f} mW "
          f"({power_spread_pct:.0f}% of mean) -- vs. the single-seed figure currently in the log (156mW)")

    if plateau_spread >= 20:
        print("\nLarge seed-to-seed spread, same order of magnitude as the INT8 check's 28-iteration spread --")
        print("this confirms the SAME fragility applies to the number currently written into")
        print("Compute/end_to_end_compute_output.log as authoritative. 434 FFTs/156W/312.4x (seed=0) is one")
        print("sample from a wide distribution, not a settled figure -- report a range or a seed-averaged")
        print("value instead of a single seed before this goes back into the document.")
    else:
        print("\nSpread is small relative to the INT8 check's 28-iteration swing -- this number looks more")
        print("stable than the plateau-iteration side-finding in the INT8 test was. Still worth reporting")
        print("as a range across seeds rather than a single-seed point estimate.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        ax.plot(range(1, len(r["history"]) + 1), r["history"], label=f"seed={r['seed']}", alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title("10.1c convergence curve across seeds (full res, real content, smooth_cutoff)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("end_to_end_seed_check.png", dpi=130)
    print("Saved plot: end_to_end_seed_check.png")


if __name__ == "__main__":
    main()
