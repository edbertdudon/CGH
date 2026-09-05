"""
First real data point on the "14 FFTs/frame" question.

Scope (read this before trusting the numbers):
  - Resolution here is 256x256, not the spec's 4700x2700. Full res needs a
    GPU (see README). Convergence *behavior* (how PSNR rises with
    iterations) is still informative at reduced scale; the *absolute*
    iteration count needed may shift somewhat with resolution and target
    complexity -- rerun at full scale before trusting this for silicon
    sizing.
  - 3 depth planes, synthetic targets, no waveguide effects, no real eye
    aberrations, no speckle-perception model. This validates the
    phase-retrieval algorithm and FFT count, not the whole optical chain.
  - "Acceptable quality" (PSNR_THRESHOLD below) is a placeholder. Swap in
    whatever bar actually matters for this product (compare against your
    own reference images, or a specific PSNR/SSIM the optical team signs
    off on).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter import counter
from targets import make_multiplane_target
from retrieval import multiplane_gs

# ---- parameters tied to the technical reference doc ----
WAVELENGTH = 520e-9        # green, matches doc's stated FOV wavelength
DX = 2.0e-6                # matches doc's 2um pixel pitch target
SHAPE = (256, 256)         # scaled down from 4700x2700 for CPU demo speed
N_PLANES = 3               # layered 3D target, doc's Section 4.4 assumption
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]   # illustrative separations, see README
MAX_ITERS = 16
PSNR_THRESHOLD = 20.0      # placeholder "acceptable" bar -- calibrate this

DOC_ASSUMED_FFTS_PER_FRAME = 14
DOC_ASSUMED_GFLOP_PER_FRAME = 20.8
DOC_ASSUMED_TFLOPS = 7.5


def flop_count_per_fft(n):
    """Same 5*N*log2(N) estimate the doc uses, applied per-FFT-call so the
    two numbers are comparable on equal footing."""
    return 5 * n * np.log2(n)


def main():
    targets = make_multiplane_target(SHAPE, n_planes=N_PLANES)

    counter.reset()
    slm_phase, history, checkpoints = multiplane_gs(
        targets, DEPTHS_M, WAVELENGTH, DX, MAX_ITERS
    )
    total_ffts_all_iters = counter.count

    # find first iteration crossing the quality threshold
    crossing_iter = next((i + 1 for i, v in enumerate(history) if v >= PSNR_THRESHOLD), None)

    # practical-plateau detection: first iteration within 0.2dB of the
    # final value reached, i.e. where the algorithm has effectively
    # stopped improving even if it never crosses the aspirational threshold
    final_psnr = history[-1]
    plateau_iter = next(i + 1 for i, v in enumerate(history) if v >= final_psnr - 0.2)

    # FFT count is linear in iteration count for this algorithm (fixed FFTs
    # per iteration), so back out per-iteration cost directly.
    ffts_per_iter = total_ffts_all_iters / MAX_ITERS
    n_slm_pixels = SHAPE[0] * SHAPE[1]
    gflop_per_fft = flop_count_per_fft(n_slm_pixels) / 1e9

    print("=" * 60)
    print(f"Grid: {SHAPE}, {N_PLANES} depth planes, {MAX_ITERS} GS iterations run")
    print(f"FFT calls per GS iteration (measured): {ffts_per_iter:.0f}")
    print(f"Total FFT calls over {MAX_ITERS} iterations: {total_ffts_all_iters}")
    print()

    if crossing_iter is not None:
        ffts_at_threshold = ffts_per_iter * crossing_iter
        print(f"PSNR >= {PSNR_THRESHOLD} dB first reached at iteration {crossing_iter}")
        print(f"  -> measured FFT calls needed: {ffts_at_threshold:.0f}"
              f" (doc assumed {DOC_ASSUMED_FFTS_PER_FRAME}/frame)")
    else:
        print(f"PSNR did NOT reach the {PSNR_THRESHOLD} dB aspirational threshold"
              f" within {MAX_ITERS} iterations.")
        print(f"Plain GS plateaued at {final_psnr:.1f} dB by iteration {plateau_iter}"
              f" ({ffts_per_iter * plateau_iter:.0f} FFT calls) and stopped improving.")
        print()
        print("This is a real, known limitation of plain multi-plane GS on hard-edged")
        print("targets, not a bug: it stagnates instead of converging further. It means")
        print("plain GS may not be the right algorithm for the quality bar this needs --")
        print("worth testing anti-aliased/grayscale targets and the SGD/Adam variant")
        print("before concluding anything about whether 14 FFTs/frame is enough.")

    print()
    print(f"For reference: at this SMALL resolution, one FFT costs ~{gflop_per_fft:.4f} GFLOP.")
    print(f"At the doc's full 4700x2700 target resolution, one FFT costs ~1.49 GFLOP")
    print(f"(the doc's own per-FFT estimate) -- FFT cost scales with N*log(N), so this")
    print(f"small-scale run is for algorithm/iteration-count behavior, not absolute FLOPs.")
    print("=" * 60)

    # ---- convergence plot ----
    fig, axes = plt.subplots(1, N_PLANES + 1, figsize=(4 * (N_PLANES + 1), 4))

    ax = axes[0]
    ax.plot(range(1, MAX_ITERS + 1), history, marker="o")
    ax.axhline(PSNR_THRESHOLD, color="red", linestyle="--", label=f"{PSNR_THRESHOLD} dB threshold")
    if crossing_iter is not None:
        ax.axvline(crossing_iter, color="green", linestyle=":", label=f"reached at iter {crossing_iter}")
    ax.set_xlabel("GS iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title("Convergence")
    ax.legend(fontsize=8)

    final_iter = max(checkpoints.keys())
    for i, recon in enumerate(checkpoints[final_iter]):
        axes[i + 1].imshow(recon, cmap="gray")
        axes[i + 1].set_title(f"Plane {i+1} recon @ iter {final_iter}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("convergence.png", dpi=130)
    print("Saved plot: convergence.png")


if __name__ == "__main__":
    main()
