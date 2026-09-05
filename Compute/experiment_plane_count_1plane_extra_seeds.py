"""
Follow-up to experiment_plane_count_seed_check.py: the 3-seed result
showed 1-plane's own seed-to-seed spread (6.86-8.51 dB, 1.65 dB) is
wider than the entire gap between 1-plane's and 2-plane's averages
(7.53 vs 7.41, 0.12 dB) -- and a pairwise check confirmed the ordering
doesn't hold up: 2-plane actually beats 1-plane in 6 of 9 seed pairings,
opposite the average's apparent direction. Before concluding anything
about the 1-vs-2-plane step, this adds 2 more seeds (3, 4) to the
1-plane condition only (2-plane and 3-plane's spreads were already tight
and consistent, no follow-up needed there).

Run on your 3060 (expect ~2 min):
    python3 experiment_plane_count_1plane_extra_seeds.py
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
EXTRA_SEEDS = [3, 4]


def main():
    print(f"Device: {DEVICE}")
    targets = make_realistic_multiplane_target(SHAPE)
    near = targets[0]

    _ = multiplane_gs_torch([near], [DEPTHS_M[0]], WAVELENGTH, DX, 1, device=DEVICE, pad_factor=PAD_FACTOR)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    rows, cols = sparse_content_bounds(near)
    results = [6.86, 7.23, 8.51]  # seeds 0,1,2 from the original run
    for seed in EXTRA_SEEDS:
        _, history, recon_by_checkpoint = multiplane_gs_torch(
            [near], [DEPTHS_M[0]], WAVELENGTH, DX, N_ITERS_BUDGET,
            device=DEVICE, seed=seed, pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        plateau_iter, reference, still_rising = find_plateau_robust(history)
        final_checkpoint = max(recon_by_checkpoint.keys())
        recon = recon_by_checkpoint[final_checkpoint][0]
        q = psnr_intensity(recon[rows, cols], near[rows, cols])
        results.append(q)
        print(f"seed={seed}: corrected={q:.2f} dB, plateau={plateau_iter}"
              f"{' STILL RISING' if still_rising else ''}")

    print(f"\nAll 5 seeds: {[round(v,2) for v in results]}")
    print(f"Average: {sum(results)/len(results):.2f} dB, range: {min(results):.2f}-{max(results):.2f}")

    two_plane_seeds = [7.30, 7.45, 7.48]
    wins = sum(1 for x in results for y in two_plane_seeds if x > y)
    total = len(results) * len(two_plane_seeds)
    print(f"\n1-plane > 2-plane pairwise (5 seeds x 3 seeds): {wins}/{total}")


if __name__ == "__main__":
    main()
