"""
10.8, last software-only item: replace Section 4.4's 5*N*log2(N) analytic
FFT-flop estimate with a real, measured figure.

Honest complication, dealt with in two layers rather than assumed away:
most profilers -- including torch.profiler -- don't have a registered
FLOP formula for FFT specifically (they're built mainly around matmul/
conv-style ops with well-defined multiply-add counts). So:

  Layer 1: try torch.profiler(with_flops=True) directly. If it reports
  nonzero FLOPs for the FFT ops, that's the real hardware-agnostic count
  we want -- use it.

  Layer 2: regardless of whether Layer 1 works, measure real GPU wall-
  clock time per FFT via torch.cuda.Event (precise GPU-side timing, not
  time.time()). Combine that measured time with the document's own
  analytic per-FFT flop formula to get an ACHIEVED GFLOP/s figure --
  not purely theoretical anymore, since the time component is now real,
  measured, on-hardware. Compare against the RTX 3060's rated 13 TFLOPS
  to see what fraction of peak this workload actually achieves, and
  against the 360Hz frame budget directly.

A true hardware-performance-counter FLOP count (via NVIDIA Nsight
Compute / ncu) would be the fully rigorous next step beyond this, but
needs elevated profiling permissions not guaranteed to be available --
flagged as optional, not attempted here.

Run on your 3060:
    python3 experiment_profiler_gflop.py
"""
import math
import torch
from torch.profiler import profile, ProfilerActivity

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)   # doc target resolution (N_y, N_x)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 16

DOC_ASSUMED_GFLOP_PER_FFT = 1.49   # Section 4.4's estimate, at full (unpadded) target resolution
DOC_ASSUMED_TFLOPS = 7.5
GPU_PEAK_TFLOPS = 13.0             # RTX 3060, FP32, from datasheet -- for utilization comparison only


def analytic_gflop_per_fft(n_pixels):
    """The same 5*N*log2(N) formula Section 4.4 uses, applied to whatever
    array size is actually being FFT'd (the padded size, not the raw
    target resolution) -- so this is the correct per-call cost to compare
    against measured timing, not the doc's headline number directly."""
    return 5 * n_pixels * math.log2(n_pixels) / 1e9


def main():
    print(f"Device: {DEVICE}")
    if DEVICE != "cuda":
        print("No GPU found -- this needs your 3060 for meaningful timing numbers.")

    targets = make_realistic_multiplane_target(SHAPE)

    # warm-up (CUDA context / cuFFT plan-building, not timed)
    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE, pad_factor=PAD_FACTOR,
                             smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # ---- Layer 1: torch.profiler with_flops ----
    print("\nLayer 1: torch.profiler(with_flops=True)...")
    counter.reset()
    activities = [ProfilerActivity.CPU]
    if DEVICE == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_flops=True) as prof:
        _, hist, cps = multiplane_gs_torch(
            targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR,
            smooth_cutoff=True
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    ffts_this_run = counter.count

    total_flops = sum(
        (evt.flops or 0) for evt in prof.key_averages() if evt.flops is not None
    )
    print(f"  Reported total FLOPs across {N_ITERS} iterations ({ffts_this_run} FFT calls): "
          f"{total_flops/1e9:.3f} GFLOP")
    if total_flops == 0:
        print("  -> Zero, as expected: FFT has no registered flop formula in torch.profiler.")
        print("     Falling back to Layer 2 (measured timing + analytic formula) below.")
    else:
        print(f"  -> Nonzero! Per-FFT: {total_flops/1e9/ffts_this_run:.4f} GFLOP "
              f"(compare to doc's assumed {DOC_ASSUMED_GFLOP_PER_FFT} GFLOP/FFT)")

    print("\n  Top ops by CUDA time (where is the time actually going):")
    sort_key = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=12))

    # ---- Layer 2: precise GPU-timed wall-clock, combined with analytic formula ----
    print("\nLayer 2: precise GPU-timed wall-clock (torch.cuda.Event)...")
    if DEVICE == "cuda":
        counter.reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _, hist2, cps2 = multiplane_gs_torch(
            targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR,
            smooth_cutoff=True
        )
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        ffts2 = counter.count
        ms_per_fft = elapsed_ms / ffts2
        ms_per_iter = elapsed_ms / N_ITERS

        padded_pixels = (SHAPE[0] * PAD_FACTOR) * (SHAPE[1] * PAD_FACTOR)
        gflop_per_fft_analytic = analytic_gflop_per_fft(padded_pixels)
        achieved_gflops_per_s = gflop_per_fft_analytic / (ms_per_fft / 1000)
        achieved_tflops = achieved_gflops_per_s / 1000
        utilization = achieved_tflops / GPU_PEAK_TFLOPS * 100

        print(f"  Total: {elapsed_ms:.2f}ms, {ffts2} FFT calls, {ms_per_fft:.4f}ms/FFT, "
              f"{ms_per_iter:.2f}ms/iteration")
        print(f"  Analytic cost per FFT at this (padded) size: {gflop_per_fft_analytic:.3f} GFLOP")
        print(f"  -> Achieved throughput: {achieved_tflops:.2f} TFLOP/s "
              f"({utilization:.1f}% of this GPU's {GPU_PEAK_TFLOPS} TFLOP/s rated peak)")
        print()
        frame_budget_ms = 1000 / 360
        print(f"  360Hz frame budget: {frame_budget_ms:.2f}ms/frame. One GS iteration measured "
              f"{ms_per_iter:.2f}ms on this GPU ({ms_per_iter/frame_budget_ms:.1f}x the budget).")
        print("  This is a consumer GPU, not the target ASIC -- not a feasibility verdict, but a")
        print("  real-hardware sanity check on whether the compute demand is in a sane ballpark.")
    else:
        print("  Skipped -- needs CUDA for meaningful GPU timing.")

    print("\n" + "=" * 70)
    print("SUMMARY -- numbers to bring back to Section 4.4/4.5:")
    if total_flops > 0:
        print(f"  Profiler-measured: {total_flops/1e9/ffts_this_run:.4f} GFLOP/FFT (direct)")
    if DEVICE == "cuda":
        print(f"  Measured-time + analytic formula: {gflop_per_fft_analytic:.3f} GFLOP/FFT, "
              f"achieved {achieved_tflops:.2f} TFLOP/s on this GPU ({utilization:.1f}% of peak)")
    print(f"  Document's current assumption: {DOC_ASSUMED_GFLOP_PER_FFT} GFLOP/FFT, "
          f"{DOC_ASSUMED_TFLOPS} TFLOP/s target")
    print("=" * 70)


if __name__ == "__main__":
    main()
