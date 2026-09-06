"""
Follow-up to experiment_subaperture_vs_discrete_tile.py. The first test
used a narrow eyebox range (4 tiles spaced 2.5px apart, far-plane-
equivalent) -- an easy case for discrete tiling, since each tile only
has to cover a small sliver of viewpoint territory. That test found no
clean win for the joint (subaperture-sampled) approach, and also found
the "tile_flicker" metric (averaging all 4 tiles' reconstructions) was
contaminated by a generic speckle-averaging effect unrelated to
viewpoint correctness (confirmed directly in experiment_flicker_
diagnostic.py: the same boost appears even averaging 4 patterns trained
on the IDENTICAL viewpoint). tile_flicker is kept here for reference
but is not used to judge the outcome -- only the clean, uncontaminated
joint-vs-tile_best comparison is.

This tests the claim that actually matters: as the eyebox range widens
while the tile BUDGET stays fixed (still just 4 discrete patterns, the
same frame-rate cost as before), does discrete-tile quality degrade
faster than the joint approach's? The paper's real claimed advantage is
a large eyebox WITHOUT a proportional increase in frame-rate cost -- a
test that keeps the range narrow can't distinguish that from "discrete
tiling was already fine here." This widens the range 2x (MAX_U_FAR_PX:
5 -> 10, so per-tile spacing doubles from 2.5px to 5px) while holding
N_TILES=4 and the training budget fixed, isolating range width as the
one changed variable.

Run on your 3060 (expect ~1 min):
    python3 experiment_subaperture_wide_eyebox.py
"""
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, sparse_content_bounds
from retrieval_torch import _to_tensor_targets
from propagation_torch import angular_spectrum_propagate, safe_abs
from metrics import psnr_intensity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
Z_FAR = DEPTHS_M[-1]
PARALLAX_RATIOS = [Z_FAR / z for z in DEPTHS_M]

MAX_U_FAR_PX = 10.0          # 2x wider than the first test (was 5.0)
N_TILES = 4                  # same budget as the first test
TILE_FRACTIONS = [-0.75, -0.25, 0.25, 0.75]
N_TEST_VIEWPOINTS = 15
N_STEPS = 500                # same training budget as the first test, deliberately not increased
BATCH_SIZE = 4
LR = 0.02
SEED = 0


def pad_crop_shift_subpixel(plane, dx_px, dy_px=0.0):
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
    return [pad_crop_shift_subpixel(p, u_far_px * ratio) for p, ratio in zip(base_planes, PARALLAX_RATIOS)]


def masked_quality(recon_planes_np, targets_np):
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
    return [safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR)) for z in DEPTHS_M]


def train_fixed_viewpoint(base_planes, u_far_px, seed, n_steps=N_STEPS):
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
    print(f"MAX_U_FAR_PX = {MAX_U_FAR_PX} (was 5.0 in the first test -- range is 2x wider)")
    t_start = time.time()
    base_planes = make_realistic_multiplane_target(SHAPE)

    tile_centers = [f * MAX_U_FAR_PX for f in TILE_FRACTIONS]
    test_viewpoints = list(np.linspace(-MAX_U_FAR_PX, MAX_U_FAR_PX, N_TEST_VIEWPOINTS))
    print(f"Tile centers: {[round(c,2) for c in tile_centers]} (spacing {tile_centers[1]-tile_centers[0]:.2f}px, "
          f"was {2.5:.2f}px in the first test)")

    print(f"\nTraining {N_TILES} discrete-tile patterns...")
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

        recon_joint = [r.detach().cpu().numpy() for r in propagate_all_planes(joint_phase)]
        q_joint = masked_quality(recon_joint, target_np)

        nearest_idx = int(np.argmin([abs(u - c) for c in tile_centers]))
        recon_best = [r.detach().cpu().numpy() for r in propagate_all_planes(tile_phases[nearest_idx])]
        q_best = masked_quality(recon_best, target_np)

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
              f"tile_best={results['tile_best'][-1]:.2f} dB, tile_flicker(contaminated)={results['tile_flicker'][-1]:.2f} dB")

    print("\n" + "=" * 80)
    for name, vals in results.items():
        avg = sum(vals) / len(vals)
        tag = " [speckle-averaging contaminated, not used for verdict]" if name == "tile_flicker" else ""
        print(f"{name:<14}avg={avg:.2f} dB, range={min(vals):.2f}-{max(vals):.2f} dB, std={np.std(vals):.2f} dB{tag}")
    print("=" * 80)

    joint_avg = sum(results["joint"]) / len(results["joint"])
    best_avg = sum(results["tile_best"]) / len(results["tile_best"])
    joint_std = np.std(results["joint"])
    best_std = np.std(results["tile_best"])

    # edge-vs-center degradation: compare quality at the extremes vs. the middle
    joint_edge = (results["joint"][0] + results["joint"][-1]) / 2
    joint_center = results["joint"][len(results["joint"]) // 2]
    best_edge = (results["tile_best"][0] + results["tile_best"][-1]) / 2
    best_center = results["tile_best"][len(results["tile_best"]) // 2]

    print(f"\nClean comparison (no averaging involved): joint avg={joint_avg:.2f} dB vs. tile_best avg={best_avg:.2f} dB")
    print(f"Uniformity: joint std={joint_std:.2f} dB, tile_best std={best_std:.2f} dB")
    print(f"Edge-vs-center degradation: joint {joint_center:.2f}->{ joint_edge:.2f} dB "
          f"({joint_edge-joint_center:+.2f}), tile_best {best_center:.2f}->{best_edge:.2f} dB ({best_edge-best_center:+.2f})")

    if best_avg - joint_avg < 0.3 and best_std > joint_std * 1.3:
        print("\nAt 2x the range, tile_best's advantage has shrunk and/or its uniformity is now worse than joint's --")
        print("consistent with the claimed scalability pattern (discrete tiling degrades faster as range widens).")
    elif joint_avg > best_avg:
        print("\nJoint now wins outright at this wider range -- the scalability pattern shows up clearly.")
    else:
        print("\nTile_best still wins clearly even at 2x the range -- the claimed scalability advantage doesn't show up")
        print("yet at this range/budget; a much larger range increase, more joint training steps, or both may be needed")
        print("before it appears, if it appears in this simplified (no-waveguide) test at all.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(test_viewpoints, results["joint"], marker="o", label="Joint (subaperture-sampled, 1 pattern)")
    ax.plot(test_viewpoints, results["tile_best"], marker="s", label="Discrete-tile, best-case (nearest tile)")
    ax.plot(test_viewpoints, results["tile_flicker"], marker="^", alpha=0.4,
            label="Discrete-tile, flicker (contaminated by speckle-averaging, reference only)")
    for c in tile_centers:
        ax.axvline(c, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Test viewpoint (far-plane-equivalent px)")
    ax.set_ylabel("Corrected/masked quality (dB)")
    ax.set_title(f"Wide-eyebox stress test (2x range), {SHAPE[0]}x{SHAPE[1]}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("subaperture_wide_eyebox.png", dpi=130)
    print("Saved plot: subaperture_wide_eyebox.png")


if __name__ == "__main__":
    main()
