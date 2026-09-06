"""
Re-runs the confirmed eyebox-architecture comparison (Docs Appendix
A.14.11: joint subaperture-sampled optimization vs. this project's
discrete-tile eyebox strategy, wide range, 4-tile budget) on REAL content
for the first time -- the depth-bucketed real photo + Japanese-caption
planes built by build_realistic_target.py, instead of the procedural
icon/text/photo set every prior eyebox test used.

Same pipeline as experiment_subaperture_wide_eyebox_multiseed.py: same
range (MAX_U_FAR_PX=10), same 4-tile budget, same 500-step training
budget, same Adam optimizer, same DEPTHS_M/parallax-shift model. Only the
content changes. Quality uses the corrected per-pixel content mask (see
test_realistic_reconstruction.py's masked_quality) -- the old
sparse_content_bounds bounding-box mask does not work for this content
(confirmed: covers 84-97% of the frame for these scattered, non-blob
planes), so per-pixel masking is used from the start here.

Run on your 3060 (expect ~2-3 min for 6 seeds):
    python3 experiment_eyebox_realistic_content.py
"""
import time
import numpy as np
import torch

import sys
sys.path.insert(0, r"D:\Documents\CGH\Demo")
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

MAX_U_FAR_PX = 10.0
N_TILES = 4
TILE_FRACTIONS = [-0.75, -0.25, 0.25, 0.75]
N_TEST_VIEWPOINTS = 15
N_STEPS = 500
BATCH_SIZE = 4
LR = 0.02
SEEDS = [0, 1, 2, 3, 4, 5]


def load_base_planes():
    return [
        np.load(r"D:\Documents\CGH\Translation\target_near.npy"),
        np.load(r"D:\Documents\CGH\Translation\target_mid_with_caption.npy"),
        np.load(r"D:\Documents\CGH\Translation\target_far.npy"),
    ]


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


def masked_quality(recon_planes_np, targets_np, content_thresh=0.05):
    psnrs = []
    for r, t in zip(recon_planes_np, targets_np):
        mask = t > content_thresh
        psnrs.append(psnr_intensity(r[mask], t[mask]))
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


def run_one_seed(base_planes, seed, test_viewpoints, tile_centers):
    joint_phase = train_joint_subaperture(base_planes, seed=seed * 100 + 999)
    tile_phases = [train_fixed_viewpoint(base_planes, c, seed=seed * 100 + i) for i, c in enumerate(tile_centers)]

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
        "joint_avg": float(np.mean(joint_vals)), "joint_std": float(np.std(joint_vals)),
        "best_avg": float(np.mean(best_vals)), "best_std": float(np.std(best_vals)),
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}, wide range (MAX_U_FAR_PX={MAX_U_FAR_PX}), {N_TILES}-tile budget, REAL content")
    t_start = time.time()
    base_planes = load_base_planes()
    tile_centers = [f * MAX_U_FAR_PX for f in TILE_FRACTIONS]
    test_viewpoints = list(np.linspace(-MAX_U_FAR_PX, MAX_U_FAR_PX, N_TEST_VIEWPOINTS))

    results = []
    for seed in SEEDS:
        t0 = time.time()
        r = run_one_seed(base_planes, seed, test_viewpoints, tile_centers)
        results.append(r)
        win = "joint" if r["joint_avg"] > r["best_avg"] else "tile_best"
        print(f"   seed {seed}: joint={r['joint_avg']:.2f}dB(std={r['joint_std']:.2f}), "
              f"tile_best={r['best_avg']:.2f}dB(std={r['best_std']:.2f}) -> {win} wins [{time.time()-t0:.1f}s]")

    joint_avgs = [r["joint_avg"] for r in results]
    best_avgs = [r["best_avg"] for r in results]
    joint_stds = [r["joint_std"] for r in results]
    best_stds = [r["best_std"] for r in results]
    wins = sum(1 for r in results if r["joint_avg"] > r["best_avg"])
    gap = np.mean(joint_avgs) - np.mean(best_avgs)

    print(f"\n{'='*80}\nSUMMARY across {len(SEEDS)} seeds, REAL content, wide range\n{'='*80}")
    print(f"joint:     mean_avg={np.mean(joint_avgs):.2f}dB (seed-std={np.std(joint_avgs):.2f}), "
          f"mean_within-range_std={np.mean(joint_stds):.2f}dB")
    print(f"tile_best: mean_avg={np.mean(best_avgs):.2f}dB (seed-std={np.std(best_avgs):.2f}), "
          f"mean_within-range_std={np.mean(best_stds):.2f}dB")
    print(f"joint wins in {wins}/{len(SEEDS)} seeds, mean(joint-tile_best) = {gap:+.2f}dB")
    print(f"\nFor comparison, procedural-content result at this exact condition (A.14.11): "
          f"joint=7.30dB, tile_best=7.28dB, gap=+0.01dB, joint wins 4/6")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
