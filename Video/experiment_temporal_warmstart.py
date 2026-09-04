"""
Section 10 next step: temporal warm-starting across frames.

Every convergence measurement so far (10.1, 10.1c, 10.2, 10.3, 10.6, 10.7,
10.7b) solves each frame from scratch, starting from a random phase. This
tests a different question: if content between adjacent frames is similar
(as it typically is at 360Hz), does seeding frame N+1's GS optimization
from frame N's *converged* phase reach a given quality in fewer
iterations than starting from random noise?

Two conditions, same synthetic frame sequence, same propagation model,
same iteration budget per frame:

  A) Cold-start every frame  -- the assumption baked into every prior
     measurement in this document, including the 58-iteration / 696-FFT
     convergence point in 10.1c.
  B) Warm-start every frame (after the first) from the previous frame's
     converged phase.

Motion is synthetic, not real video: each frame is the same realistic
multi-plane target (near/mid/far, from targets.py) shifted by a small
constant pixel offset per frame via torch.roll, approximating slow
panning/head-motion content. This is a stand-in for real motion, not a
validated content-motion model -- flagged here so it isn't mistaken for
one later.

Small scale first (512x512), same discipline as every other algorithm
test in this project (see experiment_neural_prior.py). Only worth a
full-resolution rerun (4,700x2,700, real content per 10.6) if this shows
a real, reproducible gap between cold and warm.

Requires the init_phase= patch to multiplane_gs_torch in retrieval_torch.py
(5-line diff, see chat) -- this script raises a clear error if that
hasn't been applied yet, rather than silently falling back to cold-start
behavior.

Run on your 3060:
    python3 experiment_temporal_warmstart.py
"""
import inspect
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
SHAPE = (512, 512)                   # small scale first, matches this project's usual pattern
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_BUDGET = 30                  # generous per-frame budget; we care about WHEN it crosses threshold, not just the final value
N_FRAMES = 8
SHIFT_PX_PER_FRAME = 3               # synthetic motion: constant pixel shift per frame
QUALITY_THRESHOLD_MARGIN_DB = 0.5    # "converged" = within this margin of that frame's own cold-start final PSNR
PLANE_NAMES = ["near", "mid", "far"]


def check_patch_applied():
    sig = inspect.signature(multiplane_gs_torch)
    if "init_phase" not in sig.parameters:
        raise RuntimeError(
            "multiplane_gs_torch has no `init_phase` parameter yet -- apply the "
            "5-line patch to retrieval_torch.py before running this experiment "
            "(adds an optional warm-start phase input, default None, fully "
            "backward-compatible with every existing result in this project)."
        )


def make_frame_sequence(base_planes, n_frames, shift_px):
    """
    base_planes: list of 2D arrays (near/mid/far) from
    make_realistic_multiplane_target. Returns a list of length n_frames,
    each itself a list of 3 planes, shifted by shift_px * frame_index.
    Synthetic motion stand-in -- see module docstring.
    """
    frames = []
    for f in range(n_frames):
        shifted = [torch.roll(torch.as_tensor(p), shifts=f * shift_px, dims=1).numpy() for p in base_planes]
        frames.append(shifted)
    return frames


def first_crossing(history, threshold):
    """First 1-indexed iteration at which history reaches >= threshold, or None."""
    for i, v in enumerate(history, start=1):
        if v >= threshold:
            return i
    return None


def run_sequence(frames, warm_start):
    """
    Runs GS across the frame sequence. If warm_start, seeds each frame
    after the first from the previous frame's converged phase; otherwise
    every frame starts fresh (seeded by frame index for reproducibility).
    """
    results = []
    prev_phase = None
    for f, planes in enumerate(frames):
        counter.reset()
        init_phase = prev_phase if (warm_start and prev_phase is not None) else None
        t0 = time.time()
        phase, history, _ = multiplane_gs_torch(
            planes, DEPTHS_M, WAVELENGTH, DX, N_ITERS_BUDGET,
            device=DEVICE, seed=f, pad_factor=PAD_FACTOR, init_phase=init_phase
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        results.append({"frame": f, "history": history, "ffts": counter.count, "elapsed": elapsed, "phase": phase})
        prev_phase = phase
    return results


def main():
    print(f"Device: {DEVICE}")
    check_patch_applied()

    base = make_realistic_multiplane_target(SHAPE)
    frames = make_frame_sequence(base, N_FRAMES, SHIFT_PX_PER_FRAME)

    print(f"\nA) COLD-START baseline across {N_FRAMES} frames "
          f"(every frame from random phase -- matches every prior measurement in this project)...")
    cold = run_sequence(frames, warm_start=False)

    print(f"\nB) WARM-START across {N_FRAMES} frames "
          f"(frame 0 cold, frames 1+ seeded from the previous frame's converged phase)...")
    warm = run_sequence(frames, warm_start=True)

    # Threshold = each frame's own cold-start final PSNR minus a small margin --
    # asks "how many iterations to reach essentially the quality cold-start
    # eventually reaches," the fair comparison, not an arbitrary fixed dB value.
    print("\n" + "=" * 78)
    print(f"{'Frame':<8}{'Cold final':<14}{'Cold cross-iter':<18}{'Warm cross-iter':<18}{'Speedup':<10}")
    speedups = []
    for f in range(N_FRAMES):
        cold_final = cold[f]["history"][-1]
        threshold = cold_final - QUALITY_THRESHOLD_MARGIN_DB
        cold_cross = first_crossing(cold[f]["history"], threshold)
        warm_cross = first_crossing(warm[f]["history"], threshold)
        if cold_cross and warm_cross:
            speedup = cold_cross / warm_cross
            speedups.append(speedup)
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "n/a"
        print(f"{f:<8}{cold_final:<14.2f}{str(cold_cross):<18}{str(warm_cross):<18}{speedup_str:<10}")
    print("=" * 78)

    if len(speedups) >= 2:
        # Frame 0 is cold under both conditions by construction -- excluded
        # from the average so it reflects the actual warm-start effect, not
        # a trivially-identical first frame.
        steady_state = speedups[1:]
        avg_speedup = sum(steady_state) / len(steady_state) if steady_state else speedups[0]
        print(f"\nAverage iteration speedup from warm-starting (frames 1+): {avg_speedup:.2f}x")
        if avg_speedup > 1.3:
            print("Real reduction in iterations-to-quality from warm-starting. Worth rerunning at")
            print("full resolution (4,700x2,700, real content per 10.6) before updating Gap 2's")
            print("compute figure -- this synthetic constant-shift motion is a stand-in, not a")
            print("validated content-motion model.")
        else:
            print("No meaningful speedup at this scale/motion pattern. Worth checking whether a")
            print("larger or smaller shift-per-frame changes this before ruling the idea out.")
    else:
        print(f"\nNot enough frames crossed the threshold within the iteration budget to compare --")
        print(f"consider raising N_ITERS_BUDGET (currently {N_ITERS_BUDGET}).")

    mid_f = N_FRAMES // 2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, N_ITERS_BUDGET + 1), cold[mid_f]["history"], label=f"Cold-start (frame {mid_f})", marker="o")
    ax.plot(range(1, N_ITERS_BUDGET + 1), warm[mid_f]["history"], label=f"Warm-start (frame {mid_f})", marker="s")
    ax.axhline(cold[mid_f]["history"][-1] - QUALITY_THRESHOLD_MARGIN_DB, color="gray", linestyle="--",
               label="Quality threshold")
    ax.set_xlabel("GS iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"Cold vs. warm-start convergence, frame {mid_f}, {SHAPE[0]}x{SHAPE[1]}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("temporal_warmstart_comparison.png", dpi=130)
    print("\nSaved plot: temporal_warmstart_comparison.png")


if __name__ == "__main__":
    main()
