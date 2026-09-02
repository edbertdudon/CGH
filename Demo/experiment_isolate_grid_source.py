"""
Diagnostic: WHERE does the grid artifact enter the pipeline? Padding,
aspect ratio, precision, and the frequency-domain cutoff have all been
ruled out. Two remaining candidates, tested here directly instead of
guessed at:

  A) Pure propagation math, no GS, no target at all -- propagate a
     single random-phase field once to each depth and look at the raw
     output. If the grid is already there, it's in propagation.py's
     math itself (independent of GS or target content).

  B) Full GS, but against a target with essentially no sharp edges
     (very heavy blur, near-Gaussian blob) instead of the disc/ring/
     checker shapes used so far. If the grid disappears here, it points
     to an interaction between GS's amplitude-constraint step and
     target edge content, not the propagation math.

Also included for direct comparison: the standard run (current targets,
sigma=8) you already have data for.

Run on your 3060:
    python3 experiment_isolate_grid_source.py
"""
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from propagation_torch import angular_spectrum_propagate
from targets import make_multiplane_target
from retrieval_torch import multiplane_gs_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 2700)
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
PAD_FACTOR = 2.0
N_ITERS = 16


def main():
    print(f"Device: {DEVICE}")

    # ---- A: pure random phase, single propagation, no GS, no target ----
    print("A) Propagating pure random phase (no GS, no target)...")
    torch.manual_seed(0)
    random_phase = torch.rand(SHAPE, device=DEVICE) * 2 * math.pi - math.pi
    random_field = torch.exp(1j * random_phase)
    random_recons = []
    for z in DEPTHS_M:
        out = angular_spectrum_propagate(random_field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR)
        random_recons.append(out.abs().detach().cpu().numpy())

    # ---- B: full GS against a near-featureless (heavily blurred) target ----
    print("B) Running GS against a heavily blurred (near-edgeless) target...")
    smooth_targets = make_multiplane_target(SHAPE, n_planes=3, soft=True, sigma=40.0)
    _, hist_smooth, cps_smooth = multiplane_gs_torch(
        smooth_targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR
    )
    final_smooth = max(cps_smooth.keys())

    # ---- reference: standard targets (sigma=8), same as prior runs ----
    print("Reference) Standard targets (sigma=8), for side-by-side comparison...")
    standard_targets = make_multiplane_target(SHAPE, n_planes=3, soft=True, sigma=8.0)
    _, hist_std, cps_std = multiplane_gs_torch(
        standard_targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR
    )
    final_std = max(cps_std.keys())

    print()
    print("Check the saved image: does the grid appear in row A (pure propagation,")
    print("no target involved at all)? If yes, it's in the propagation math, full stop --")
    print("nothing about GS or targets is relevant. If row A is clean but row B (smooth")
    print("target) still shows the grid, it's coming from GS's amplitude-constraint step")
    print("interacting with target content. If row B is ALSO clean, the hard edges in the")
    print("original disc/ring/checker targets are the source.")

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(3):
        axes[0, i].imshow(random_recons[i], cmap="gray")
        axes[0, i].set_title(f"A: pure propagation, depth {i+1}")
        axes[0, i].axis("off")
        axes[1, i].imshow(cps_smooth[final_smooth][i], cmap="gray")
        axes[1, i].set_title(f"B: GS, near-edgeless target, plane {i+1}")
        axes[1, i].axis("off")
        axes[2, i].imshow(cps_std[final_std][i], cmap="gray")
        axes[2, i].set_title(f"Ref: GS, standard target, plane {i+1}")
        axes[2, i].axis("off")
    plt.tight_layout()
    plt.savefig("isolate_grid_source.png", dpi=130)
    print("Saved plot: isolate_grid_source.png")


if __name__ == "__main__":
    main()
