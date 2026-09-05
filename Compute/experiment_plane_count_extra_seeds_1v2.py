"""
Follow-up to experiment_plane_count_seed_check.py + the 1-plane-only
extra-seeds run: the 1-vs-2-plane step came out as a near-coin-flip
across seed pairings (9/15, 60%) even after adding 2 extra seeds to the
1-plane condition alone. Before accepting or rejecting the intuitive
"more simultaneous constraints should hurt monotonically" prior, this
adds 2 more seeds to BOTH the 1-plane and 2-plane conditions (bringing
them to 7 and 5 seeds respectively) to see whether the ordering settles
with more data, in either direction.

Run on your 3060 (expect ~5-8 min):
    python3 experiment_plane_count_extra_seeds_1v2.py
"""
import torch
from targets import make_realistic_multiplane_target, sparse_content_bounds
from retrieval_torch import multiplane_gs_torch
from metrics import psnr_intensity
from convergence import find_plateau_robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_BUDGET = 300
EXTRA_SEEDS = [5, 6]

# Previously measured (existing runs, not repeated here)
ONE_PLANE_EXISTING = [6.86, 7.23, 8.51, 8.12, 8.42]     # seeds 0-4
TWO_PLANE_EXISTING = [7.30, 7.45, 7.48]                  # seeds 0-2


def score_condition(targets_subset, depths_subset, plane_global_idxs, seed):
    _, history, recon_by_checkpoint = multiplane_gs_torch(
        targets_subset, depths_subset, WAVELENGTH, DX, N_ITERS_BUDGET,
        device=DEVICE, seed=seed, pad_factor=PAD_FACTOR, smooth_cutoff=True
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    plateau_iter, reference, still_rising = find_plateau_robust(history)
    final_checkpoint = max(recon_by_checkpoint.keys())
    recon_planes = recon_by_checkpoint[final_checkpoint]
    psnrs = []
    for r, t, gidx in zip(recon_planes, targets_subset, plane_global_idxs):
        if gidx in (0, 1):  # near, mid -- sparse, mask to content bounds
            rows, cols = sparse_content_bounds(t)
            psnrs.append(psnr_intensity(r[rows, cols], t[rows, cols]))
        else:
            psnrs.append(psnr_intensity(r, t))
    avg = sum(psnrs) / len(psnrs)
    return avg, plateau_iter, still_rising


def main():
    print(f"Device: {DEVICE}")
    targets = make_realistic_multiplane_target(SHAPE)
    near, mid = targets[0], targets[1]

    _ = multiplane_gs_torch([near], [DEPTHS_M[0]], WAVELENGTH, DX, 1, device=DEVICE, pad_factor=PAD_FACTOR)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    one_plane_new = []
    print("\n1-plane (near), extra seeds:")
    for seed in EXTRA_SEEDS:
        avg, plateau, rising = score_condition([near], [DEPTHS_M[0]], [0], seed)
        one_plane_new.append(avg)
        print(f"   seed={seed}: {avg:.2f} dB, plateau={plateau}{' STILL RISING' if rising else ''}")

    two_plane_new = []
    print("\n2-planes (near+mid), extra seeds:")
    for seed in EXTRA_SEEDS:
        avg, plateau, rising = score_condition([near, mid], [DEPTHS_M[0], DEPTHS_M[1]], [0, 1], seed)
        two_plane_new.append(avg)
        print(f"   seed={seed}: {avg:.2f} dB, plateau={plateau}{' STILL RISING' if rising else ''}")

    one_all = ONE_PLANE_EXISTING + one_plane_new
    two_all = TWO_PLANE_EXISTING + two_plane_new

    print(f"\n1-plane, all {len(one_all)} seeds: {[round(v,2) for v in one_all]}")
    print(f"   avg={sum(one_all)/len(one_all):.2f} dB, range={min(one_all):.2f}-{max(one_all):.2f}")
    print(f"2-planes, all {len(two_all)} seeds: {[round(v,2) for v in two_all]}")
    print(f"   avg={sum(two_all)/len(two_all):.2f} dB, range={min(two_all):.2f}-{max(two_all):.2f}")

    wins = sum(1 for x in one_all for y in two_all if x > y)
    total = len(one_all) * len(two_all)
    print(f"\n1-plane > 2-plane pairwise ({len(one_all)}x{len(two_all)}): {wins}/{total} ({100*wins/total:.0f}%)")

    if wins / total >= 0.75:
        print("Clearly settled: 1-plane reliably beats 2-plane -- the intuitive monotonic")
        print("ordering is confirmed for this step too.")
    elif wins / total <= 0.25:
        print("Clearly settled the OTHER way: 2-plane reliably beats 1-plane -- the intuitive")
        print("ordering does NOT hold for this step; report the reversal directly.")
    else:
        print("Still not settled even with more seeds -- this step is genuinely too close to")
        print("call, not just under-sampled. Report as statistically indistinguishable.")


if __name__ == "__main__":
    main()
