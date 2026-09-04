"""
Builds the single clean, full-resolution, real-content, end-to-end
compute measurement to eventually replace the "combined from two
separate tests" estimate currently in Section 10.1b / Gap 2.

Everything from ONE run this time, not stitched together from different
tests under different conditions:
  - Full target resolution (4,700x2,700)
  - Real content (near/mid/far, from 10.6)
  - smooth_cutoff=True (the validated bug fix from 10.1b -- ~32% faster,
    confirmed identical PSNR to the hard cutoff)
  - A genuine convergence sweep (100 iterations, not a fixed cutoff) so
    the FFT count needed is read directly off this run's own plateau,
    not assumed from a small-scale test
  - Real GPU-timed wall-clock (torch.cuda.Event) from this same run

Does not assume a target PSNR -- Section 10.8 flagged that as the one
item that needs a human, not more simulation. This reports the full
convergence curve so whatever target eventually gets set can be read
off it directly, plus the compute implied by wherever it actually
plateaus in the meantime.

This is a longer run than previous experiments -- expect a few minutes,
not seconds (100 iterations x 12 FFTs/iteration = 1,200 FFT calls).

Run on your 3060:
    python3 experiment_end_to_end_compute.py
"""
import math
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch
from metrics import psnr_intensity as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)   # doc target resolution (N_y, N_x)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 100           # long sweep for a real convergence curve, not a guess
PLATEAU_TOLERANCE_DB = 0.2

GPU_PEAK_TFLOPS = 13.0  # RTX 3060 FP32, for utilization context only
TOPS_PER_W = 6.5        # same BFP16 efficiency assumption as Section 4.5
POWER_BUDGET_MW = 500

DOC_FFTS_ASSUMED = 14
DOC_GFLOP_PER_FFT_ASSUMED = 1.49


def analytic_gflop_per_fft(n_pixels):
    return 5 * n_pixels * math.log2(n_pixels) / 1e9


def main():
    print(f"Device: {DEVICE}")
    if DEVICE != "cuda":
        print("No GPU found -- this needs your 3060 for a meaningful result.")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    print(f"Running {N_ITERS}-iteration convergence sweep "
          f"(full resolution, real content, smooth_cutoff, single run)...")
    counter.reset()
    if DEVICE == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    t0 = time.time()

    phase, history, cps = multiplane_gs_torch(
        targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE,
        pad_factor=PAD_FACTOR, smooth_cutoff=True
    )

    if DEVICE == "cuda":
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        elapsed_ms = (time.time() - t0) * 1000
    total_ffts = counter.count

    print(f"Done: {elapsed_ms/1000:.1f}s total, {total_ffts} FFT calls, "
          f"final quality {history[-1]:.2f} dB")

    # ---- convergence analysis: read plateau directly off this run ----
    final_psnr = history[-1]
    plateau_iter = next(i + 1 for i, v in enumerate(history) if v >= final_psnr - PLATEAU_TOLERANCE_DB)
    still_rising = (final_psnr - history[max(0, N_ITERS - 10)]) > PLATEAU_TOLERANCE_DB
    ffts_per_iter = total_ffts / N_ITERS
    ffts_at_plateau = ffts_per_iter * plateau_iter

    print(f"\nPlateaued (within {PLATEAU_TOLERANCE_DB} dB of final) at iteration "
          f"{plateau_iter}/{N_ITERS} ({history[plateau_iter-1]:.2f} dB)")
    if still_rising:
        print("WARNING: still measurably rising over the last 10 iterations -- consider")
        print("rerunning with a higher N_ITERS before treating this plateau as final.")
    print(f"FFTs per iteration (measured, this run): {ffts_per_iter:.0f}")
    print(f"FFTs needed to reach plateau: {ffts_at_plateau:.0f}")

    # ---- real per-FFT cost, from THIS run's own timing ----
    ms_per_fft = elapsed_ms / total_ffts
    padded_pixels = (SHAPE[0] * PAD_FACTOR) * (SHAPE[1] * PAD_FACTOR)
    gflop_per_fft = analytic_gflop_per_fft(padded_pixels)
    achieved_tflops = gflop_per_fft / (ms_per_fft / 1000) / 1000
    utilization = achieved_tflops / GPU_PEAK_TFLOPS * 100

    print(f"\nMeasured this run: {ms_per_fft:.3f} ms/FFT")
    print(f"Analytic cost per FFT at this padded size: {gflop_per_fft:.2f} GFLOP")
    print(f"Achieved throughput: {achieved_tflops:.2f} TFLOP/s ({utilization:.1f}% of 3060 peak)")

    # ---- end-to-end compute/power implied, from this single run ----
    gflop_per_frame = ffts_at_plateau * gflop_per_fft
    tflops_needed = gflop_per_frame * 360 / 1000
    power_mw = tflops_needed / TOPS_PER_W * 1000
    budget_ratio = power_mw / POWER_BUDGET_MW

    doc_gflop_per_frame = DOC_FFTS_ASSUMED * DOC_GFLOP_PER_FFT_ASSUMED

    print("\n" + "=" * 70)
    print("END-TO-END RESULT (single run -- full resolution, real content,")
    print("no stitching together of separate tests):")
    print(f"  {ffts_at_plateau:.0f} FFTs/frame x {gflop_per_fft:.2f} GFLOP/FFT = "
          f"{gflop_per_frame:.1f} GFLOP/frame")
    print(f"  x 360Hz = {tflops_needed:.2f} TFLOP/s -> {power_mw:.0f}mW at {TOPS_PER_W} TOPS/W "
          f"-> {budget_ratio:.1f}x over the {POWER_BUDGET_MW}mW budget")
    print(f"  (Section 4.4's original assumption: {DOC_FFTS_ASSUMED} FFTs x "
          f"{DOC_GFLOP_PER_FFT_ASSUMED} GFLOP = {doc_gflop_per_frame:.1f} GFLOP/frame, "
          f"7.5 TFLOP/s, 1,154mW, 2.3x)")
    print(f"  (Previous stitched-together estimate: ~546 GFLOP/frame, ~30W, ~60x)")
    if DEVICE == "cuda":
        print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("=" * 70)
    print()
    print("Quality target still unresolved (10.8) -- this run reports the plateau it")
    print("actually reached, not a pass/fail against a threshold. Once a defensible PSNR")
    print("target exists, read the required iteration count directly off the saved curve.")

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(1, N_ITERS + 1), history)
    axes[0].axhline(final_psnr - PLATEAU_TOLERANCE_DB, color="red", linestyle="--",
                     label=f"plateau threshold ({PLATEAU_TOLERANCE_DB} dB)")
    axes[0].axvline(plateau_iter, color="green", linestyle=":",
                     label=f"plateau at iter {plateau_iter}")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean PSNR across planes (dB)")
    axes[0].set_title(f"Full-resolution convergence, real content, smooth_cutoff\n"
                       f"({SHAPE[1]}x{SHAPE[0]}, {N_ITERS} iterations)")
    axes[0].legend()

    final_iter = max(cps.keys())
    axes[1].axis("off")
    axes[1].text(0.05, 0.9, "Result summary:", fontsize=12, fontweight="bold", transform=axes[1].transAxes)
    summary_lines = [
        f"Final quality: {final_psnr:.2f} dB",
        f"Plateau: iter {plateau_iter}, {ffts_at_plateau:.0f} FFTs/frame",
        f"Measured: {ms_per_fft:.3f} ms/FFT, {achieved_tflops:.2f} TFLOP/s achieved",
        f"Implied: {gflop_per_frame:.1f} GFLOP/frame, {tflops_needed:.2f} TFLOP/s @ 360Hz",
        f"Implied power: {power_mw:.0f}mW ({budget_ratio:.1f}x the {POWER_BUDGET_MW}mW budget)",
    ]
    for i, line in enumerate(summary_lines):
        axes[1].text(0.05, 0.75 - i * 0.12, line, fontsize=10, transform=axes[1].transAxes)
    plt.tight_layout()
    plt.savefig("end_to_end_compute_convergence.png", dpi=130)
    print("Saved plot: end_to_end_compute_convergence.png")

    fig2, axarr = plt.subplots(1, 3, figsize=(12, 4))
    for i, name in enumerate(["near", "mid", "far"]):
        axarr[i].imshow(cps[final_iter][i], cmap="gray")
        axarr[i].set_title(f"{name}, iter {final_iter} ({psnr_np(cps[final_iter][i], targets[i]):.1f} dB)")
        axarr[i].axis("off")
    plt.tight_layout()
    plt.savefig("end_to_end_compute_final_recon.png", dpi=130)
    print("Saved plot: end_to_end_compute_final_recon.png")


if __name__ == "__main__":
    main()
