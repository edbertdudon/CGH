"""
Multi-seed confirmation of experiment_subaperture_wide_eyebox.py.

The single-seed wide-range test (MAX_U_FAR_PX=10.0, 2x the first test's
range, same 4-tile budget, same 500-step training budget) found joint
beating tile_best on average and on uniformity: joint avg=7.33dB (std=0.09),
tile_best avg=7.24dB (std=0.16, with a real sawtooth pattern -- peaks near
tile centers, dips at tile-boundary midpoints). That's a reversal from the
narrow-range test (joint 7.39 vs tile_best 7.55, tile_best winning) --
consistent with the paper's claimed scalability pattern, but it's a single
random seed, so it could just be luck of the initial phase draw rather
than a real, repeatable effect.

This does not change the eyebox range, tile budget, training budget, or
content -- ONLY the seed controlling initial phase and (for joint) the
per-step random viewpoint sampling -- and repeats the entire narrow-vs-wide
comparison across several seeds, so we can see the seed-to-seed spread
and check whether joint's win at the wide range holds up or was a fluke.

Run on your 3060 (expect ~6-8 min for 6 seeds):
    python3 experiment_subaperture_wide_eyebox_multiseed.py
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

N_TILES = 4
TILE_FRACTIONS = [-0.75, -0.25, 0.25, 0.75]
N_TEST_VIEWPOINTS = 15
N_STEPS = 500
BATCH_SIZE = 4
LR = 0.02
SEEDS = [0, 1, 2, 3, 4, 5]

RANGES = {
    "narrow (5.0px, first test)": 5.0,
    "wide (10.0px, 2x, follow-up)": 10.0,
}


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


def train_joint_subaperture(base_planes, max_u_far_px, seed, n_steps=N_STEPS, batch_size=BATCH_SIZE):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    slm_phase = (torch.rand(SHAPE, device=DEVICE) * 2 * np.pi - np.pi).requires_grad_(True)
    opt = torch.optim.Adam([slm_phase], lr=LR)
    for step in range(n_steps):
        us = rng.uniform(-max_u_far_px, max_u_far_px, size=batch_size)
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


def run_one(base_planes, max_u_far_px, seed):
    tile_centers = [f * max_u_far_px for f in TILE_FRACTIONS]
    test_viewpoints = list(np.linspace(-max_u_far_px, max_u_far_px, N_TEST_VIEWPOINTS))

    tile_phases = []
    for i, center in enumerate(tile_centers):
        # offset tile seeds by seed*100 so different top-level seeds don't collide
        phase = train_fixed_viewpoint(base_planes, center, seed=seed * 100 + i)
        tile_phases.append(phase)

    joint_phase = train_joint_subaperture(base_planes, max_u_far_px, seed=seed * 100 + 999)

    joint_vals, best_vals = [], []
    for u in test_viewpoints:
        target_np = viewpoint_targets(base_planes, u)

        recon_joint = [r.detach().cpu().numpy() for r in propagate_all_planes(joint_phase)]
        q_joint = masked_quality(recon_joint, target_np)

        nearest_idx = int(np.argmin([abs(u - c) for c in tile_centers]))
        recon_best = [r.detach().cpu().numpy() for r in propagate_all_planes(tile_phases[nearest_idx])]
        q_best = masked_quality(recon_best, target_np)

        joint_vals.append(sum(q_joint) / len(q_joint))
        best_vals.append(sum(q_best) / len(q_best))

    return {
        "joint_avg": float(np.mean(joint_vals)),
        "joint_std": float(np.std(joint_vals)),
        "best_avg": float(np.mean(best_vals)),
        "best_std": float(np.std(best_vals)),
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    t_start = time.time()
    base_planes = make_realistic_multiplane_target(SHAPE)

    all_results = {}
    for range_name, max_u in RANGES.items():
        print(f"\n{'='*80}\nRange: {range_name}\n{'='*80}")
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            r = run_one(base_planes, max_u, seed)
            per_seed.append(r)
            win = "joint" if r["joint_avg"] > r["best_avg"] else "tile_best"
            print(f"   seed {seed}: joint={r['joint_avg']:.2f}dB (std={r['joint_std']:.2f}), "
                  f"tile_best={r['best_avg']:.2f}dB (std={r['best_std']:.2f}) -> {win} wins "
                  f"[{time.time()-t0:.1f}s]")
        all_results[range_name] = per_seed

    print(f"\n{'='*80}\nSUMMARY across {len(SEEDS)} seeds\n{'='*80}")
    for range_name, per_seed in all_results.items():
        joint_avgs = [r["joint_avg"] for r in per_seed]
        best_avgs = [r["best_avg"] for r in per_seed]
        joint_stds = [r["joint_std"] for r in per_seed]
        best_stds = [r["best_std"] for r in per_seed]
        wins = sum(1 for r in per_seed if r["joint_avg"] > r["best_avg"])
        print(f"\n{range_name}:")
        print(f"   joint:     mean_avg={np.mean(joint_avgs):.2f}dB (seed-to-seed std={np.std(joint_avgs):.2f}), "
              f"mean_within-range_std={np.mean(joint_stds):.2f}dB")
        print(f"   tile_best: mean_avg={np.mean(best_avgs):.2f}dB (seed-to-seed std={np.std(best_avgs):.2f}), "
              f"mean_within-range_std={np.mean(best_stds):.2f}dB")
        print(f"   joint wins on avg quality in {wins}/{len(SEEDS)} seeds")
        diff = np.mean(joint_avgs) - np.mean(best_avgs)
        print(f"   mean(joint_avg - tile_best_avg) = {diff:+.2f}dB")

    narrow_key = "narrow (5.0px, first test)"
    wide_key = "wide (10.0px, 2x, follow-up)"
    narrow_diff = np.mean([r["joint_avg"] for r in all_results[narrow_key]]) - \
        np.mean([r["best_avg"] for r in all_results[narrow_key]])
    wide_diff = np.mean([r["joint_avg"] for r in all_results[wide_key]]) - \
        np.mean([r["best_avg"] for r in all_results[wide_key]])
    wide_wins = sum(1 for r in all_results[wide_key] if r["joint_avg"] > r["best_avg"])

    print(f"\n{'='*80}")
    print(f"Narrow range: mean(joint-tile_best) = {narrow_diff:+.2f}dB")
    print(f"Wide range:   mean(joint-tile_best) = {wide_diff:+.2f}dB")
    print(f"Joint wins the wide range in {wide_wins}/{len(SEEDS)} seeds")
    if wide_diff > 0 and wide_wins >= len(SEEDS) - 1 and wide_diff > narrow_diff + 0.2:
        print("\nCONFIRMED across seeds: joint's advantage at the wider range is consistent, not a single-seed fluke.")
        print("The scalability pattern (discrete tiling degrades faster than joint as range widens, fixed tile budget)")
        print("replicates across independent random initializations.")
    elif wide_wins <= 1:
        print("\nNOT CONFIRMED: the single-seed wide-range result does not replicate -- joint's apparent win was")
        print("likely a fluke of that one initialization, not a real, repeatable effect. Treat the original")
        print("single-seed wide-range finding as unconfirmed / withdrawn.")
    else:
        print("\nMIXED: joint wins more often than not at the wide range, but not consistently enough across seeds")
        print("to call this a confirmed, repeatable effect yet -- report as a genuine but noisy/marginal signal.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SEEDS))
    width = 0.35
    narrow_joint = [r["joint_avg"] for r in all_results[narrow_key]]
    narrow_best = [r["best_avg"] for r in all_results[narrow_key]]
    wide_joint = [r["joint_avg"] for r in all_results[wide_key]]
    wide_best = [r["best_avg"] for r in all_results[wide_key]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (name, jv, bv) in zip(axes, [("Narrow (5.0px)", narrow_joint, narrow_best),
                                          ("Wide (10.0px)", wide_joint, wide_best)]):
        ax.bar(x - width/2, jv, width, label="joint")
        ax.bar(x + width/2, bv, width, label="tile_best")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in SEEDS])
        ax.set_xlabel("seed")
        ax.set_title(name)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Avg quality (dB)")
    plt.suptitle(f"Multi-seed joint vs. tile_best, narrow vs. wide eyebox range ({len(SEEDS)} seeds)")
    plt.tight_layout()
    plt.savefig("subaperture_wide_eyebox_multiseed.png", dpi=130)
    print("Saved plot: subaperture_wide_eyebox_multiseed.png")


if __name__ == "__main__":
    main()
