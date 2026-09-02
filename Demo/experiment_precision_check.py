"""
Diagnostic: is the grid/crosshair artifact seen at full resolution caused
by single-precision (complex64) floating-point round-off accumulating in
larger FFTs? Aspect ratio and padding have both been ruled out already
(same artifact persists at 2700x2700 square, pad_factor=2.0) -- this is
the next most likely explanation, since round-off grows with FFT size
and tends to concentrate along rows/columns (matching the axis-aligned
pattern seen).

Runs at 2700x2700 (not the full 4700x2700) on purpose: consumer GPUs like
the 3060 are deliberately much slower at double precision (often
1/32-1/64 of single-precision speed), so this checks the hypothesis at a
size you already have a single-precision baseline for, before committing
to a slow full-resolution double-precision run.

Run on your 3060:
    python3 experiment_precision_check.py
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_multiplane_target
from retrieval_torch import multiplane_gs_torch
from metrics import psnr as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 2700)
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 16
PAD_FACTOR = 2.0


def main():
    print(f"Device: {DEVICE}")
    targets = make_multiplane_target(SHAPE, n_planes=3, soft=True, sigma=8.0)

    print("Running single precision (complex64)...")
    phase_32, hist_32, cps_32 = multiplane_gs_torch(
        targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR,
        complex_dtype=torch.complex64
    )
    final_32 = max(cps_32.keys())
    q32 = [psnr_np(cps_32[final_32][i], targets[i]) for i in range(3)]
    print(f"  quality: {[round(q,1) for q in q32]}")

    print("Running double precision (complex128) -- this will be noticeably slower...")
    phase_64, hist_64, cps_64 = multiplane_gs_torch(
        targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR,
        complex_dtype=torch.complex128
    )
    final_64 = max(cps_64.keys())
    q64 = [psnr_np(cps_64[final_64][i], targets[i]) for i in range(3)]
    print(f"  quality: {[round(q,1) for q in q64]}")

    print()
    print("If the grid artifact is precision-related, the double-precision images below")
    print("should look visibly cleaner (less axis-aligned striping) than single precision,")
    print("even if the PSNR numbers themselves are similar.")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        axes[0, i].imshow(cps_32[final_32][i], cmap="gray")
        axes[0, i].set_title(f"complex64, plane {i+1} ({q32[i]:.1f} dB)")
        axes[0, i].axis("off")
        axes[1, i].imshow(cps_64[final_64][i], cmap="gray")
        axes[1, i].set_title(f"complex128, plane {i+1} ({q64[i]:.1f} dB)")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("precision_check.png", dpi=130)
    print("Saved plot: precision_check.png")


if __name__ == "__main__":
    main()
