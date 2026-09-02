"""
Run this on your machine (the 3060), not in a sandbox -- needs a real
CUDA install. This is NOT executable in the environment that built it
(no GPU there), so it hasn't been run end-to-end. It has been written to
mirror propagation.py / retrieval.py as closely as possible, which WERE
validated on CPU. If something breaks, the error message plus which line
it's on will tell us fast where the torch port diverges.

What this does:
  1. Sanity check: run GS on GPU, confirm it reproduces the CPU plateau
     behavior (~11-13 dB with these synthetic targets) -- confirms the
     torch port is correct before trusting anything else.
  2. Run SGD/Adam on the same targets, same FFT-call budget, and compare
     directly: which algorithm gets further, and with how many FFTs.
  3. Time it, so you get a real wall-clock number to sanity-check against
     the doc's 7.5 TFLOP/s target -- not proof the ASIC can do it (see
     README), but a first-order check that the algorithm isn't wildly
     more expensive than assumed.

Install (once): pip install torch --index-url https://download.pytorch.org/whl/cu121
(check https://pytorch.org/get-started/locally/ for the CUDA version that
matches your driver)
"""
import time
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_multiplane_target
from retrieval_torch import multiplane_gs_torch, multiplane_sgd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)          # bump toward (2700, 4700) once this runs clean
N_PLANES = 3
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_GS = 16              # GS plateaus fast -- 16 is already past its wall
N_STEPS_SGD = 500            # SGD is still improving at 100; give it more room
DOC_ASSUMED_FFTS_PER_FRAME = 14


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
    print(f"Device: {DEVICE}" + ("" if DEVICE == "cuda" else "  (no GPU found -- running on CPU, will be slow)"))
    targets = make_multiplane_target(SHAPE, n_planes=N_PLANES, soft=True, sigma=3.0)

    (phase_gs, hist_gs, cps_gs), t_gs, ffts_gs = run_and_time(
        multiplane_gs_torch, targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS_GS, device=DEVICE
    )
    (phase_sgd, hist_sgd, cps_sgd, _), t_sgd, ffts_sgd = run_and_time(
        multiplane_sgd, targets, DEPTHS_M, WAVELENGTH, DX, N_STEPS_SGD, device=DEVICE
    )

    print("=" * 60)
    print(f"GS:  {N_ITERS_GS} iters, {ffts_gs} FFT calls (real, no instrumentation overhead), "
          f"{t_gs:.3f}s, final PSNR {hist_gs[-1]:.1f} dB")
    print(f"SGD: {N_STEPS_SGD} steps, {ffts_sgd} FFT calls (forward only -- backward()")
    print(f"     does its own FFTs internally, invisible to the counter; true cost is")
    print(f"     roughly ~2x this number), {t_sgd:.3f}s, final PSNR {hist_sgd[-1]:.1f} dB")
    print(f"(doc assumed {DOC_ASSUMED_FFTS_PER_FRAME} FFTs/frame total, for comparison)")
    print()
    if hist_sgd[-1] > hist_gs[-1] + 1.0:
        print("SGD is meaningfully ahead of GS -- worth switching the compute estimate")
        print("in Section 4.4 to an SGD-based count instead of GS.")
    elif hist_gs[-1] > hist_sgd[-1] + 1.0:
        print("GS is still ahead at this budget.")
    else:
        print("Roughly comparable right now -- check whether SGD's curve has flattened")
        print("yet (look at the plot) before concluding anything either way.")

    if DEVICE == "cuda":
        frame_budget_s = 1.0 / 360  # doc's 360Hz composite target
        print()
        print(f"Wall-clock per frame at this resolution: GS {t_gs/N_ITERS_GS*1000:.2f}ms/iter, "
              f"SGD {t_sgd/N_STEPS_SGD*1000:.2f}ms/step")
        print(f"360Hz frame budget: {frame_budget_s*1000:.2f}ms. This is {SHAPE}, not the full "
              f"4700x2700 -- treat as a rough scaling signal, not a frame-rate proof.")
    print("=" * 60)

    # ---- convergence curves ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, N_ITERS_GS + 1), hist_gs, marker="o", label="GS")
    ax.plot(range(1, N_STEPS_SGD + 1), hist_sgd, marker=".", markersize=3, label="SGD/Adam")
    ax.set_xlabel("Iteration / step")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"GS vs SGD convergence, {SHAPE[0]}x{SHAPE[1]}, {N_PLANES} planes")
    ax.legend()
    plt.tight_layout()
    plt.savefig("gs_vs_sgd_convergence.png", dpi=130)
    print("Saved plot: gs_vs_sgd_convergence.png")

    # ---- what the reconstructions actually look like at the final checkpoint ----
    final_gs = max(cps_gs.keys())
    final_sgd = max(cps_sgd.keys())
    fig2, axes = plt.subplots(2, N_PLANES, figsize=(4 * N_PLANES, 8))
    for i in range(N_PLANES):
        axes[0, i].imshow(cps_gs[final_gs][i], cmap="gray")
        axes[0, i].set_title(f"GS plane {i+1} @ iter {final_gs} ({hist_gs[-1]:.1f} dB)")
        axes[0, i].axis("off")
        axes[1, i].imshow(cps_sgd[final_sgd][i], cmap="gray")
        axes[1, i].set_title(f"SGD plane {i+1} @ step {final_sgd} ({hist_sgd[-1]:.1f} dB)")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("gs_vs_sgd_reconstructions.png", dpi=130)
    print("Saved plot: gs_vs_sgd_reconstructions.png")


if __name__ == "__main__":
    main()
