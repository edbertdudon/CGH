"""
Follow-up to experiment_subaperture_wide_eyebox_multiseed.py. That test
confirmed a real uniformity advantage for the joint approach at a wide
eyebox range (10px) with a 4-tile budget, but found NO confirmed average-
quality advantage (6-seed mean gap +0.01dB, noise-level) -- a much more
mixed result than the original single-seed test suggested.

This tests whether a scarcer tile budget over the SAME wide range makes
the average-quality gap show up for real. The paper's actual claimed
advantage is about a FIXED compute/frame-rate budget covering MORE
territory -- our 4-tile test at 10px range may simply not have stressed
discrete-tile hard enough (4 tiles across 10px is still fairly dense).
Halving the tile budget to 2 (same 10px range, same 500-step training
budget per pattern) makes each tile responsible for 2x the territory,
which should make discrete-tile's coverage problem much worse if the
claim is real -- while the joint pattern is trained identically to
before (it never had a "tile budget" concept in the first place).

Isolates tile budget as the ONE changed variable: range, content, training
steps, optimizer, and seeds are all unchanged from the confirmed wide-
range condition. The joint pattern is trained once per seed and reused
across both tile-budget conditions (its training procedure doesn't
depend on the tile budget at all), so this only spends extra compute on
training the 2-tile condition's phases.

Run on your 3060 (expect ~10-12 min for 6 seeds x 2 tile budgets):
    python3 experiment_subaperture_fewer_tiles_multiseed.py
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

MAX_U_FAR_PX = 10.0  # wide range only -- already confirmed, held fixed here
TILE_BUDGETS = {
    4: [-0.75, -0.25, 0.25, 0.75],   # confirmed condition, repeated for a clean in-script comparison
    2: [-0.5, 0.5],                   # scarcer budget, same range -- the new condition
}
N_TEST_VIEWPOINTS = 15
N_STEPS = 500
BATCH_SIZE = 4
LR = 0.02
SEEDS = [0, 1, 2, 3, 4, 5]


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


def run_one_seed(base_planes, seed, test_viewpoints):
    joint_phase = train_joint_subaperture(base_planes, MAX_U_FAR_PX, seed=seed * 100 + 999)
    recon_joint = [r.detach().cpu().numpy() for r in propagate_all_planes(joint_phase)]
    joint_vals = []
    for u in test_viewpoints:
        target_np = viewpoint_targets(base_planes, u)
        q_joint = masked_quality(recon_joint, target_np)
        joint_vals.append(sum(q_joint) / len(q_joint))

    per_budget = {}
    for n_tiles, fractions in TILE_BUDGETS.items():
        centers = [f * MAX_U_FAR_PX for f in fractions]
        tile_phases = [train_fixed_viewpoint(base_planes, c, seed=seed * 100 + i) for i, c in enumerate(centers)]
        best_vals = []
        for u in test_viewpoints:
            target_np = viewpoint_targets(base_planes, u)
            nearest_idx = int(np.argmin([abs(u - c) for c in centers]))
            recon_best = [r.detach().cpu().numpy() for r in propagate_all_planes(tile_phases[nearest_idx])]
            q_best = masked_quality(recon_best, target_np)
            best_vals.append(sum(q_best) / len(q_best))
        per_budget[n_tiles] = {
            "best_avg": float(np.mean(best_vals)),
            "best_std": float(np.std(best_vals)),
        }

    return {
        "joint_avg": float(np.mean(joint_vals)),
        "joint_std": float(np.std(joint_vals)),
        "tiles": per_budget,
    }


def main():
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}, wide range only (MAX_U_FAR_PX={MAX_U_FAR_PX}), tile budgets: {list(TILE_BUDGETS.keys())}")
    t_start = time.time()
    base_planes = make_realistic_multiplane_target(SHAPE)
    test_viewpoints = list(np.linspace(-MAX_U_FAR_PX, MAX_U_FAR_PX, N_TEST_VIEWPOINTS))

    all_results = []
    for seed in SEEDS:
        t0 = time.time()
        r = run_one_seed(base_planes, seed, test_viewpoints)
        all_results.append(r)
        tiles_str = ", ".join(f"{n}-tile={v['best_avg']:.2f}dB(std={v['best_std']:.2f})" for n, v in r["tiles"].items())
        print(f"   seed {seed}: joint={r['joint_avg']:.2f}dB(std={r['joint_std']:.2f}), {tiles_str} [{time.time()-t0:.1f}s]")

    print(f"\n{'='*80}\nSUMMARY across {len(SEEDS)} seeds (wide range, {MAX_U_FAR_PX}px)\n{'='*80}")
    joint_avgs = [r["joint_avg"] for r in all_results]
    print(f"joint: mean_avg={np.mean(joint_avgs):.2f}dB (seed-to-seed std={np.std(joint_avgs):.2f})")

    for n_tiles in TILE_BUDGETS:
        best_avgs = [r["tiles"][n_tiles]["best_avg"] for r in all_results]
        best_stds = [r["tiles"][n_tiles]["best_std"] for r in all_results]
        wins = sum(1 for r in all_results if r["joint_avg"] > r["tiles"][n_tiles]["best_avg"])
        gap = np.mean(joint_avgs) - np.mean(best_avgs)
        print(f"\n{n_tiles}-tile budget:")
        print(f"   tile_best: mean_avg={np.mean(best_avgs):.2f}dB (seed-to-seed std={np.std(best_avgs):.2f}), "
              f"mean_within-range_std={np.mean(best_stds):.2f}dB")
        print(f"   joint wins in {wins}/{len(SEEDS)} seeds, mean(joint-tile_best) = {gap:+.2f}dB")

    gap_4 = np.mean(joint_avgs) - np.mean([r["tiles"][4]["best_avg"] for r in all_results])
    gap_2 = np.mean(joint_avgs) - np.mean([r["tiles"][2]["best_avg"] for r in all_results])
    print(f"\n{'='*80}")
    print(f"4-tile budget: mean(joint-tile_best) = {gap_4:+.2f}dB")
    print(f"2-tile budget: mean(joint-tile_best) = {gap_2:+.2f}dB")
    if gap_2 > gap_4 + 0.2 and gap_2 > 0.15:
        print("\nCONFIRMED: a scarcer tile budget over the same range DOES surface a real average-quality")
        print("advantage for joint -- the claim needed a harder budget/coverage mismatch to show up in software.")
    elif gap_2 > gap_4 + 0.1:
        print("\nPARTIAL: the gap widens in joint's favor with a scarcer budget, but not by enough yet to call")
        print("a confirmed average-quality win -- a still-scarcer budget or wider range may be needed.")
    else:
        print("\nNOT CONFIRMED: halving the tile budget over the same range did not meaningfully change the")
        print("average-quality picture -- the uniformity-vs-average-quality split found at 4 tiles appears to")
        print("hold at 2 tiles as well, not just an artifact of that specific budget.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SEEDS))
    width = 0.25
    joint_vals = [r["joint_avg"] for r in all_results]
    tiles4_vals = [r["tiles"][4]["best_avg"] for r in all_results]
    tiles2_vals = [r["tiles"][2]["best_avg"] for r in all_results]
    ax.bar(x - width, joint_vals, width, label="joint")
    ax.bar(x, tiles4_vals, width, label="tile_best (4 tiles)")
    ax.bar(x + width, tiles2_vals, width, label="tile_best (2 tiles)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("seed")
    ax.set_ylabel("Avg quality (dB)")
    ax.set_title(f"Joint vs. tile_best at 4 vs. 2 tiles, wide range ({MAX_U_FAR_PX}px), {len(SEEDS)} seeds")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("subaperture_fewer_tiles_multiseed.png", dpi=130)
    print("Saved plot: subaperture_fewer_tiles_multiseed.png")


if __name__ == "__main__":
    main()
