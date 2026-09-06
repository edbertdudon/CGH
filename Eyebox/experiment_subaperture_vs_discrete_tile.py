"""
First test of the eyebox-architecture idea flagged in 10.11/A.14.9: does
jointly optimizing ONE phase pattern against a randomly-sampled continuum
of eyebox viewpoints (the core algorithmic idea behind Choi et al. 2025's
synthetic-aperture holography) produce smoother, more uniform coverage
across the eyebox than this project's current discrete-tile approach
(N independent phase patterns, one per fixed tile center, cycled at N x
frame rate)?

IMPORTANT SIMPLIFICATION, stated plainly per this project's standing
practice: the real paper's eyebox expansion comes from a physically
steered aperture through a real waveguide, modeled with a learned,
camera-calibrated partially-coherent propagation model -- none of that
is replicated here (no waveguide, no MEMS, no partial coherence; this
project has never modeled any of those, per every prior revision's
stated caveat). What IS tested is the narrower algorithmic question this
project's own simulation tools CAN address: given a viewpoint-dependent
target (see below), does a jointly-trained single pattern generalize
across viewpoints better than several independently-trained patterns
that each only ever see one fixed viewpoint during training?

Viewpoint-dependent target, approximated without a real light-field
renderer: this project's existing near/mid/far layers are shifted
sideways by an amount proportional to 1/depth (nearer planes shift more
per unit eyebox displacement than farther ones) -- the same physically-
motivated parallax principle already used for temporal motion in
Video/experiment_independent_content.py's make_parallax_motion_sequence
(there: motion over time; here: displacement over eyebox position).
This is a simplification of a true 4D light field (a real 3D scene
rendered from many camera positions), not the genuine article -- but it
is a real, non-trivial viewpoint-dependence test, not a straw man.

Two conditions, same optimizer (Adam/SGD-style throughout, isolating
the objective structure as the only variable, not conflating it with a
different base optimizer):
  A) Discrete-tile baseline: N_TILES=4 independent phase patterns, each
     trained against ONE fixed viewpoint's shifted targets only --
     mirrors this project's actual 2x2 eyebox-multiplexing scheme
     (Section 4.3), scored two ways: "best-case" (always shown its own
     assigned pattern) and "flicker-realistic" (all 4 patterns'
     reconstructed amplitudes averaged, mimicking what an eye actually
     integrates when all 4 are cycled faster than it can resolve,
     regardless of the eye's true position).
  B) Joint subaperture-sampled: ONE phase pattern, each training step
     scored against a small random BATCH of continuous viewpoints
     sampled from the full range, not just N_TILES fixed centers --
     directly mirrors equation (5) in Choi et al. 2025.

Both evaluated on a dense, held-out sweep of test viewpoints spanning
the same range (more points than either condition trained on).

Small scale first (512x512), matching this project's usual pattern --
a full-resolution confirmation is needed before trusting this if the
small-scale result looks promising, per this project's own repeated
lesson about small-scale findings not always surviving the jump to
target resolution.

Run on your 3060 (expect ~15-20 min):
    python3 experiment_subaperture_vs_discrete_tile.py
"""
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, sparse_content_bounds
from retrieval_torch import _to_tensor_targets, _psnr_torch
from propagation_torch import angular_spectrum_propagate, safe_abs
from metrics import psnr_intensity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
Z_FAR = DEPTHS_M[-1]
PARALLAX_RATIOS = [Z_FAR / z for z in DEPTHS_M]  # [6.0, 2.0, 1.0] -- near shifts most

MAX_U_FAR_PX = 5.0          # far-plane-equivalent max viewpoint offset, in pixels
N_TILES = 4
TILE_FRACTIONS = [-0.75, -0.25, 0.25, 0.75]
N_TEST_VIEWPOINTS = 15
N_STEPS = 500
BATCH_SIZE = 4              # random viewpoints sampled per joint training step
LR = 0.02
SEED = 0


def pad_crop_shift_subpixel(plane, dx_px, dy_px=0.0):
    """Non-wrapping shift, integer-rounded. Adapted from
    Video/experiment_independent_content.py's helper of the same name."""
    t = torch.as_tensor(plane)
    h, w = t.shape
    dx_i, dy_i = int(round(dx_px)), int(round(dy_px))
    shifted = torch.zeros_like(t)
    src_x0, src_x1 = max(0, -dx_i), w - max(0, dx_i)
    dst_x0, dst_x1 = max(0, dx_i), w - max(0, -dx_i)
    src_y0, src_y1 = max(0, -dy_i), h - max(0, dy_i)
    dst_y0, dst_y1 = max(0, dy_i), h - max(0, -dy_i)
    if src_x1 > src_x0 and src_y1 > src_y0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = t[src_y0:src_y1, src_x0:src_x1]
    return shifted.numpy()


def viewpoint_targets(base_planes, u_far_px):
    """Shift each plane by u_far_px * its parallax ratio -- near planes
    shift more than far planes for the same eyebox displacement u."""
    return [pad_crop_shift_subpixel(p, u_far_px * ratio) for p, ratio in zip(base_planes, PARALLAX_RATIOS)]


def masked_quality(recon_planes_np, targets_np):
    """psnr_intensity, masked for sparse near/mid content, whole-frame for dense far -- the
    methodology fix established during the plane-count audit, applied from the start here."""
    psnrs = []
    for i, (r, t) in enumerate(zip(recon_planes_np, targets_np)):
        if i in (0, 1):
            rows, cols = sparse_content_bounds(t)
            psnrs.append(psnr_intensity(r[rows, cols], t[rows, cols]))
        else:
            psnrs.append(psnr_intensity(r, t))
    return psnrs


def propagate_all_planes(slm_phase):
    field = torch.exp(1j * slm_phase)
    recons = []
    for z in DEPTHS_M:
        recon = safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR))
        recons.append(recon)
    return recons


def train_fixed_viewpoint(base_planes, u_far_px, seed, n_steps=N_STEPS):
    """Standard single-viewpoint SGD solve -- the discrete-tile building block."""
    targets_np = viewpoint_targets(base_planes, u_far_px)
    targets_amp = _to_tensor_targets(targets_np, DEVICE)
    torch.manual_seed(seed)
    slm_phase = (torch.rand(SHAPE, device=DEVICE) * 2 * np.pi - np.pi).requires_grad_(True)
    opt = torch.optim.Adam([slm_phase], lr=LR)
    for step in range(n_steps):
        opt.zero_grad()
        field = torch.exp(1j * slm_phase)
        loss = torch.tensor(0.0, device=DEVICE)
        for target_amp, z in zip(targets_amp, DEPTHS_M):
            recon = safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR))
            loss = loss + torch.mean((recon - target_amp) ** 2)
        loss.backward()
        opt.step()
    return slm_phase.detach()


def train_joint_subaperture(base_planes, seed, n_steps=N_STEPS, batch_size=BATCH_SIZE):
    """Joint solve: each step samples a random batch of continuous
    viewpoints and optimizes ONE pattern against all of them at once --
    mirrors equation (5)'s random sub-aperture batch sampling."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    slm_phase = (torch.rand(SHAPE, device=DEVICE) * 2 * np.pi - np.pi).requires_grad_(True)
    opt = torch.optim.Adam([slm_phase], lr=LR)
    for step in range(n_steps):
        us = rng.uniform(-MAX_U_FAR_PX, MAX_U_FAR_PX, size=batch_size)
        opt.zero_grad()
        field = torch.exp(1j * slm_phase)
        loss = torch.tensor(0.0, device=DEVICE)
        for u in us:
            targets_np = viewpoint_targets(base_planes, u)
            targets_amp = _to_tensor_targets(targets_np, DEVICE)
            for target_amp, z in zip(targets_amp, DEPTHS_M):
                recon = safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR))
                loss = loss + torch.mean((recon - target_amp) ** 2) / batch_size
        loss.backward()
        opt.step()
    return slm_phase.detach()


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    base_planes = make_realistic_multiplane_target(SHAPE)

    tile_centers = [f * MAX_U_FAR_PX for f in TILE_FRACTIONS]
    test_viewpoints = list(np.linspace(-MAX_U_FAR_PX, MAX_U_FAR_PX, N_TEST_VIEWPOINTS))

    print(f"\nTraining {N_TILES} discrete-tile patterns at centers {[round(c,2) for c in tile_centers]}...")
    tile_phases = []
    for i, center in enumerate(tile_centers):
        t0 = time.time()
        phase = train_fixed_viewpoint(base_planes, center, seed=SEED + i)
        tile_phases.append(phase)
        print(f"   tile {i} (center={center:.2f}px): {time.time()-t0:.1f}s")

    print(f"\nTraining joint subaperture-sampled pattern (batch={BATCH_SIZE}, {N_STEPS} steps)...")
    t0 = time.time()
    joint_phase = train_joint_subaperture(base_planes, seed=SEED)
    print(f"   joint: {time.time()-t0:.1f}s")

    print(f"\nEvaluating across {N_TEST_VIEWPOINTS} held-out test viewpoints...")
    results = {"joint": [], "tile_best": [], "tile_flicker": []}
    for u in test_viewpoints:
        target_np = viewpoint_targets(base_planes, u)

        # joint: single pattern, direct evaluation
        recon_joint = [r.detach().cpu().numpy() for r in propagate_all_planes(joint_phase)]
        q_joint = masked_quality(recon_joint, target_np)

        # tile-best: nearest tile center's pattern
        nearest_idx = int(np.argmin([abs(u - c) for c in tile_centers]))
        recon_best = [r.detach().cpu().numpy() for r in propagate_all_planes(tile_phases[nearest_idx])]
        q_best = masked_quality(recon_best, target_np)

        # tile-flicker: average reconstructed amplitude across all 4 tile patterns
        # (mimics temporal integration when all 4 are cycled faster than the eye resolves)
        all_tile_recons = [propagate_all_planes(p) for p in tile_phases]
        avg_recon = [
            torch.stack([all_tile_recons[t][plane_idx] for t in range(N_TILES)]).mean(dim=0).cpu().numpy()
            for plane_idx in range(3)
        ]
        q_flicker = masked_quality(avg_recon, target_np)

        results["joint"].append(sum(q_joint) / len(q_joint))
        results["tile_best"].append(sum(q_best) / len(q_best))
        results["tile_flicker"].append(sum(q_flicker) / len(q_flicker))
        print(f"   u={u:+.2f}px: joint={results['joint'][-1]:.2f} dB, "
              f"tile_best={results['tile_best'][-1]:.2f} dB, tile_flicker={results['tile_flicker'][-1]:.2f} dB")

    print("\n" + "=" * 80)
    for name, vals in results.items():
        avg = sum(vals) / len(vals)
        print(f"{name:<14}avg={avg:.2f} dB, range={min(vals):.2f}-{max(vals):.2f} dB, "
              f"std={np.std(vals):.2f} dB")
    print("=" * 80)

    joint_avg = sum(results["joint"]) / len(results["joint"])
    best_avg = sum(results["tile_best"]) / len(results["tile_best"])
    flicker_avg = sum(results["tile_flicker"]) / len(results["tile_flicker"])
    joint_std = np.std(results["joint"])
    best_std = np.std(results["tile_best"])

    print(f"\nJoint vs. tile-best (most favorable case for discrete-tile): {joint_avg:.2f} vs {best_avg:.2f} dB")
    print(f"Joint vs. tile-flicker (more realistic case for discrete-tile): {joint_avg:.2f} vs {flicker_avg:.2f} dB")
    print(f"Uniformity across viewpoints (lower std = more consistent coverage): "
          f"joint std={joint_std:.2f} dB, tile-best std={best_std:.2f} dB")

    if joint_avg > flicker_avg + 0.5 and joint_std < best_std:
        print("\nJoint approach wins on both average quality (vs. realistic tile behavior) and")
        print("uniformity across the eyebox -- worth a full-resolution confirmation.")
    elif joint_avg > flicker_avg + 0.5:
        print("\nJoint approach has higher average quality than realistic tile behavior, but not")
        print("clearly more uniform -- a partial, not complete, win.")
    else:
        print("\nNo clear win at this scale -- report plainly rather than overstating.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(test_viewpoints, results["joint"], marker="o", label="Joint (subaperture-sampled, 1 pattern)")
    ax.plot(test_viewpoints, results["tile_best"], marker="s", label="Discrete-tile, best-case (nearest tile)")
    ax.plot(test_viewpoints, results["tile_flicker"], marker="^", label="Discrete-tile, flicker-realistic (all 4 averaged)")
    for c in tile_centers:
        ax.axvline(c, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Test viewpoint (far-plane-equivalent px)")
    ax.set_ylabel("Corrected/masked quality (dB)")
    ax.set_title(f"Joint subaperture-sampled vs. discrete-tile eyebox coverage, {SHAPE[0]}x{SHAPE[1]}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("subaperture_vs_discrete_tile.png", dpi=130)
    print("Saved plot: subaperture_vs_discrete_tile.png")


if __name__ == "__main__":
    main()
