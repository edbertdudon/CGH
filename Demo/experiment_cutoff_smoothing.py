"""
Diagnostic: is the grid/crosshair artifact caused by the hard
propagating/evanescent boolean cutoff in the transfer function? Padding,
aspect ratio, and precision have all been ruled out already (artifact
persists identically at 2700x2700 square, both pad factors, both
precisions) -- this is the next candidate: a sharp discontinuity in
frequency space, invisible when frequency sampling is coarse (small
apertures) and increasingly well-resolved (hence visible) as the
aperture grows and frequency sampling gets finer.

Runs at 2700x2700, matching the precision-check baseline, single
precision (fast) since precision was already ruled out separately.

Run on your 3060:
    python3 experiment_cutoff_smoothing.py
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_multiplane_target
from retrieval_torch import multiplane_gs_torch
from metrics import psnr_intensity as psnr_np

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

    print("Running with hard cutoff (current default)...")
    _, _, cps_hard = multiplane_gs_torch(
        targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR,
        smooth_cutoff=False
    )
    final_hard = max(cps_hard.keys())
    q_hard = [psnr_np(cps_hard[final_hard][i], targets[i]) for i in range(3)]
    print(f"  quality: {[round(q,1) for q in q_hard]}")

    print("Running with smoothed cutoff...")
    _, _, cps_soft = multiplane_gs_torch(
        targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR,
        smooth_cutoff=True, cutoff_width=0.05
    )
    final_soft = max(cps_soft.keys())
    q_soft = [psnr_np(cps_soft[final_soft][i], targets[i]) for i in range(3)]
    print(f"  quality: {[round(q,1) for q in q_soft]}")

    print()
    print("If the grid artifact comes from the hard cutoff, the smoothed row below")
    print("should look visibly cleaner (less axis-aligned striping) than the hard row.")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        axes[0, i].imshow(cps_hard[final_hard][i], cmap="gray")
        axes[0, i].set_title(f"hard cutoff, plane {i+1} ({q_hard[i]:.1f} dB)")
        axes[0, i].axis("off")
        axes[1, i].imshow(cps_soft[final_soft][i], cmap="gray")
        axes[1, i].set_title(f"smooth cutoff, plane {i+1} ({q_soft[i]:.1f} dB)")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("cutoff_smoothing_check.png", dpi=130)
    print("Saved plot: cutoff_smoothing_check.png")


if __name__ == "__main__":
    main()
