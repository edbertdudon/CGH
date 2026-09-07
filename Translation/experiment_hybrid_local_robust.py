"""
Tests the third eyebox-architecture option identified after re-reading what
Choi et al.'s real system actually does: not our pure-joint extreme (one
pattern trained across the WHOLE eyebox range) and not discrete-tile (each
of 4 patterns trained for exactly ONE fixed viewpoint) -- a hybrid, closer
to their real MEMS-steered approach: a FEW discrete positions (here: 2,
half of discrete-tile's 4), each trained against a LOCAL neighborhood of
viewpoints (like joint's random sampling, but bounded to that position's
own zone) rather than either one exact point or the entire range.

Hypothesis: this should capture most of joint's no-snapping benefit (each
position is trained to blend into its own local neighborhood, not to be
sharp-then-nothing at a boundary) while keeping more of discrete-tile's
per-position quality (each pattern only has to cover half the range, not
all of it).

Same exact condition as every other eyebox test in this project (wide
range, MAX_U_FAR_PX=10, 500-step training budget, Adam, procedural
content, 512x512) so the result drops directly into the existing
3-way comparison (A.14.11 discrete-tile=7.28dB, joint=7.30dB, both
6 seeds) without needing to re-run either extreme.

Run on your 3060 (expect ~2-3 min for 6 seeds):
    python3 experiment_hybrid_local_robust.py
"""
import sys
import time
import numpy as np
import torch

sys.path.insert(0, r"D:\Documents\CGH\Demo")
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

MAX_U_FAR_PX = 10.0
N_HYBRID_POSITIONS = 2
HYBRID_CENTERS = [-5.0, 5.0]       # split the range in half
HYBRID_LOCAL_HALF_WIDTH = 5.0      # each position's own local neighborhood (contiguous, no gap)
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


def train_hybrid_position(base_planes, center, half_width, seed, n_steps=N_STEPS, batch_size=BATCH_SIZE):
    """Like joint training, but the random viewpoint sample is bounded to
    this position's own local neighborhood [center-half_width, center+half_width],
    not the whole eyebox range."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    slm_phase = (torch.rand(SHAPE, device=DEVICE) * 2 * np.pi - np.pi).requires_grad_(True)
    opt = torch.optim.Adam([slm_phase], lr=LR)
    for step in range(n_steps):
        us = rng.uniform(center - half_width, center + half_width, size=batch_size)
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
    positions = [train_hybrid_position(base_planes, c, HYBRID_LOCAL_HALF_WIDTH, seed=seed * 100 + i)
                 for i, c in enumerate(HYBRID_CENTERS)]

    vals = []
    for u in test_viewpoints:
        target_np = viewpoint_targets(base_planes, u)
        nearest_idx = int(np.argmin([abs(u - c) for c in HYBRID_CENTERS]))
        recon = [r.detach().cpu().numpy() for r in propagate_all_planes(positions[nearest_idx])]
        q = masked_quality(recon, target_np)
        vals.append(sum(q) / len(q))
    return {"avg": float(np.mean(vals)), "std": float(np.std(vals)), "per_viewpoint": vals}


def main():
    print(f"Device: {DEVICE}")
    print(f"Hybrid: {N_HYBRID_POSITIONS} positions at {HYBRID_CENTERS}, "
          f"local half-width {HYBRID_LOCAL_HALF_WIDTH}px, seeds {SEEDS}")
    t_start = time.time()
    base_planes = make_realistic_multiplane_target(SHAPE)
    test_viewpoints = list(np.linspace(-MAX_U_FAR_PX, MAX_U_FAR_PX, N_TEST_VIEWPOINTS))

    results = []
    for seed in SEEDS:
        t0 = time.time()
        r = run_one_seed(base_planes, seed, test_viewpoints)
        results.append(r)
        print(f"   seed {seed}: avg={r['avg']:.2f}dB, within-range std={r['std']:.2f}dB [{time.time()-t0:.1f}s]")

    avgs = [r["avg"] for r in results]
    stds = [r["std"] for r in results]
    print(f"\n{'='*90}")
    print(f"Hybrid ({N_HYBRID_POSITIONS} local-robust positions): mean_avg={np.mean(avgs):.2f}dB "
          f"(seed-to-seed std={np.std(avgs):.2f}), mean_within-range_std={np.mean(stds):.2f}dB")
    print(f"\nFor comparison, same exact condition, 6 seeds (A.14.11):")
    print(f"   Discrete-tile (4 positions, single-viewpoint each): 7.28dB avg, within-range std 0.17dB")
    print(f"   Joint (1 pattern, whole-range sampling):            7.30dB avg, within-range std 0.09dB")
    print(f"   Hybrid ({N_HYBRID_POSITIONS} positions, local sampling):        "
          f"{np.mean(avgs):.2f}dB avg, within-range std {np.mean(stds):.2f}dB")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
