"""
Stage 3 of the 10.5 sequence: does everything found at 256/512px hold at
the real 4,700x2,700 target resolution?

Uses GS only, not SGD -- stage 1 established SGD doesn't beat GS's
ceiling, and GS is far cheaper on GPU memory (no autograd graph to
retain), so there's no reason to fight SGD's memory footprint at full
res right now.

Padding: previous small-scale runs used pad_factor=2 to avoid FFT
wraparound, because a 256/512px aperture at 2um pitch is only
~0.5-1mm wide, and diffraction spread at these depths is a large
fraction of that. At full resolution the real aperture is ~9.4mm --
the same physical spread is a much smaller fraction of it, so less
padding is needed. Defaults to pad_factor=1.3 here; bump back to 2 if
you see edge artifacts in the output images.

Includes a GPU warm-up pass before timing anything, fixing the artifact
flagged in Section 10.3 (first-executed config absorbing CUDA/cuFFT
setup cost).

Run on your 3060:
    python3 experiment_full_resolution.py
"""
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_multiplane_target
from retrieval_torch import multiplane_gs_torch
from metrics import psnr_intensity as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 2700)         # matches doc's target (N_y, N_x)
PAD_FACTOR = 2.0             # see docstring -- lighter than the small-scale default
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 16


def report_memory(label):
    if DEVICE == "cuda":
        alloc = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [{label}] peak GPU memory so far: {alloc:.2f} GB")


def run_and_time(fn, *args, **kwargs):
    counter.reset()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    result = fn(*args, **kwargs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return result, elapsed, counter.count


def main():
    print(f"Device: {DEVICE}")
    if DEVICE != "cuda":
        print("No GPU found -- this WILL be very slow at full resolution. Consider")
        print("stopping and running on the 3060 instead.")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB total")

    targets = make_multiplane_target(SHAPE, n_planes=3, soft=True, sigma=8.0)

    # ---- warm-up: absorb CUDA context / cuFFT plan-building cost before timing ----
    print("Warming up GPU (building cuFFT plans, not timed)...")
    _ = run_and_time(
        multiplane_gs_torch, targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE, pad_factor=PAD_FACTOR
    )
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print("Warm-up done. Timings below are now trustworthy.\n")

    # ---- simultaneous ----
    (phase_sim, hist_sim, cps_sim), t_sim, ffts_sim = run_and_time(
        multiplane_gs_torch, targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR
    )
    report_memory("after simultaneous run")
    final_sim_iter = max(cps_sim.keys())
    quality_sim = [psnr_np(cps_sim[final_sim_iter][i], targets[i]) for i in range(3)]

    # ---- sequential ----
    seq_results = []
    total_ffts_seq = 0
    total_time_seq = 0.0
    for i in range(3):
        (phase_i, hist_i, cps_i), t_i, ffts_i = run_and_time(
            multiplane_gs_torch, [targets[i]], [DEPTHS_M[i]], WAVELENGTH, DX, N_ITERS,
            device=DEVICE, seed=i, pad_factor=PAD_FACTOR
        )
        final_i = max(cps_i.keys())
        q = psnr_np(cps_i[final_i][0], targets[i])
        seq_results.append({"quality": q, "ffts": ffts_i, "time": t_i, "recon": cps_i[final_i][0]})
        total_ffts_seq += ffts_i
        total_time_seq += t_i
    report_memory("after sequential runs")

    print("=" * 70)
    print(f"Resolution: {SHAPE[1]}x{SHAPE[0]} (doc target), pad_factor={PAD_FACTOR}")
    print()
    print("SIMULTANEOUS")
    for i, q in enumerate(quality_sim):
        print(f"  plane {i+1}: {q:.1f} dB")
    print(f"  total FFTs: {ffts_sim}, wall-clock: {t_sim*1000:.1f}ms (now trustworthy -- warmed up)")
    print()
    print("SEQUENTIAL")
    for i, r in enumerate(seq_results):
        print(f"  plane {i+1}: {r['quality']:.1f} dB  ({r['ffts']} FFTs, {r['time']*1000:.1f}ms)")
    print(f"  total FFTs: {total_ffts_seq}, wall-clock: {total_time_seq*1000:.1f}ms")
    print("=" * 70)

    print()
    print("Compare against small-scale (512x512) findings:")
    print("  simultaneous plateau was ~11-13 dB per plane -- does it still land there?")
    print("  sequential advantage was ~+3 dB per plane at equal FFT count -- still holds?")
    print("  small-scale wall-clock wasn't trustworthy (no warm-up); this run's numbers are.")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        axes[0, i].imshow(cps_sim[final_sim_iter][i], cmap="gray")
        axes[0, i].set_title(f"Simultaneous, plane {i+1} ({quality_sim[i]:.1f} dB)")
        axes[0, i].axis("off")
        axes[1, i].imshow(seq_results[i]["recon"], cmap="gray")
        axes[1, i].set_title(f"Sequential, plane {i+1} ({seq_results[i]['quality']:.1f} dB)")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("full_resolution_results.png", dpi=130)
    print("Saved plot: full_resolution_results.png")


if __name__ == "__main__":
    main()
