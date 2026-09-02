"""
Stage 2 of the 10.5 sequence: sequential (one depth plane per sub-frame)
vs simultaneous (all planes solved by one shared phase pattern).

This is the one with a real product tradeoff attached: sequential gets
meaningfully better quality per plane (established informally last
session), but needs 3x more sub-frames -- multiplying the existing 2x2
eyebox-steering sub-frame count (4) up to 12, pushing composite frame
rate from 360Hz to 1080Hz. This script measures the real numbers behind
that tradeoff directly, at the same 512x512 scale used in prior GPU runs.

Uses GS, not SGD -- stage 1 established tuning doesn't move the ceiling,
and GS is simpler/faster, so it isolates the architecture question
(planes-at-once vs. planes-in-sequence) without also varying algorithm.

Run on your 3060:
    python3 experiment_sequential_vs_simultaneous.py
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
from metrics import psnr as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 16          # GS plateaus well before this (stage 0 finding)

EYEBOX_SUBFRAMES = 4  # existing 2x2 eyebox steering, Section 4.3
PERCEPTUAL_HZ = 90    # Section 4.3


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
    targets = make_multiplane_target(SHAPE, n_planes=3, soft=True, sigma=3.0)

    # ---- simultaneous: one phase pattern, all 3 planes at once ----
    (phase_sim, hist_sim, cps_sim), t_sim, ffts_sim = run_and_time(
        multiplane_gs_torch, targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE
    )
    final_sim_iter = max(cps_sim.keys())
    quality_sim_per_plane = [psnr_np(cps_sim[final_sim_iter][i], targets[i]) for i in range(3)]

    # ---- sequential: 3 independent single-plane solves ----
    seq_results = []
    total_ffts_seq = 0
    total_time_seq = 0.0
    for i in range(3):
        (phase_i, hist_i, cps_i), t_i, ffts_i = run_and_time(
            multiplane_gs_torch, [targets[i]], [DEPTHS_M[i]], WAVELENGTH, DX, N_ITERS,
            device=DEVICE, seed=i
        )
        final_i = max(cps_i.keys())
        q = psnr_np(cps_i[final_i][0], targets[i])
        seq_results.append({"quality": q, "ffts": ffts_i, "time": t_i, "recon": cps_i[final_i][0]})
        total_ffts_seq += ffts_i
        total_time_seq += t_i

    print("=" * 70)
    print("SIMULTANEOUS (1 sub-frame, all 3 planes solved by one phase pattern)")
    for i, q in enumerate(quality_sim_per_plane):
        print(f"  plane {i+1}: {q:.1f} dB")
    print(f"  total FFTs: {ffts_sim}, wall-clock: {t_sim*1000:.1f}ms")
    print()
    print("SEQUENTIAL (3 sub-frames, one plane solved at a time)")
    for i, r in enumerate(seq_results):
        print(f"  plane {i+1}: {r['quality']:.1f} dB  ({r['ffts']} FFTs, {r['time']*1000:.1f}ms)")
    print(f"  total FFTs: {total_ffts_seq}, wall-clock: {total_time_seq*1000:.1f}ms")
    print("=" * 70)

    fft_ratio = total_ffts_seq / ffts_sim
    time_ratio = total_time_seq / t_sim
    avg_gain_db = float(np.mean([seq_results[i]["quality"] - quality_sim_per_plane[i] for i in range(3)]))

    print(f"Sequential costs {fft_ratio:.2f}x the FFTs and {time_ratio:.2f}x the wall-clock time")
    print(f"of simultaneous, for an average +{avg_gain_db:.1f} dB quality gain per plane.")
    print()

    seq_subframes = EYEBOX_SUBFRAMES * 3
    sim_subframes = EYEBOX_SUBFRAMES * 1
    seq_composite_hz = PERCEPTUAL_HZ * seq_subframes
    sim_composite_hz = PERCEPTUAL_HZ * sim_subframes
    print(f"Frame-budget implication (on top of existing {EYEBOX_SUBFRAMES}x eyebox steering):")
    print(f"  Simultaneous: {sim_subframes} sub-frames/perceptual-frame -> {sim_composite_hz}Hz composite (matches doc's 360Hz)")
    print(f"  Sequential:   {seq_subframes} sub-frames/perceptual-frame -> {seq_composite_hz}Hz composite "
          f"({seq_composite_hz/sim_composite_hz:.0f}x higher)")

    # ---- visualize: what each approach actually looks like, side by side ----
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        axes[0, i].imshow(cps_sim[final_sim_iter][i], cmap="gray")
        axes[0, i].set_title(f"Simultaneous, plane {i+1} ({quality_sim_per_plane[i]:.1f} dB)")
        axes[0, i].axis("off")
        axes[1, i].imshow(seq_results[i]["recon"], cmap="gray")
        axes[1, i].set_title(f"Sequential, plane {i+1} ({seq_results[i]['quality']:.1f} dB)")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("sequential_vs_simultaneous.png", dpi=130)
    print("Saved plot: sequential_vs_simultaneous.png")


if __name__ == "__main__":
    main()
