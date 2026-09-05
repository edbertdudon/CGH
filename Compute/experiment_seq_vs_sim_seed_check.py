"""
Multi-seed stress test of the sequential-vs-simultaneous depth-plane
rendering finding (originally measured in
experiment_sequential_vs_simultaneous_converged.py from a SINGLE random
seed per condition -- simultaneous used seed=0, each sequential
single-plane solve used seed=i matching its own plane index). That
single-realization result found ~0dB delta for near/mid (sparse content,
masked PSNR) and a real +2.1 to +2.8dB advantage for sequential on far
(dense photo-like content).

Given this session's repeated finding that single-seed numbers can be
unreliable (the 10.1c compute figure swung 130-282mW across seeds; the
donor-seeding result reversed once properly checked), this reruns both
conditions across 5 independent random seeds each, at full resolution,
run to a generous, confirmed-converged iteration budget -- and compares
final QUALITY (not iteration count, which is the thing already shown to
be seed-fragile; final quality has been comparatively stable across
every multi-seed check run so far this session).

Same masking convention as the original test: near (icon) and mid
(text) are sparse content and use masked PSNR (sparse_content_bounds);
far (photo-like scene) is dense and uses whole-frame PSNR.

Run on your 3060 (expect ~25-30 minutes):
    python3 experiment_seq_vs_sim_seed_check.py
"""
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, sparse_content_bounds
from retrieval_torch import multiplane_gs_torch
from metrics import psnr as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 200
N_SEEDS = 5
PLANE_NAMES = ["near", "mid", "far"]
SPARSE_PLANES = {0, 1}


def masked_or_full_psnr(recon, target, plane_idx):
    if plane_idx in SPARSE_PLANES:
        rows, cols = sparse_content_bounds(target)
        return psnr_np(recon[rows, cols], target[rows, cols])
    return psnr_np(recon, target)


def still_rising(history, tol=0.2):
    if len(history) < 20:
        return False
    return (history[-1] - sum(history[-20:-10]) / 10) > tol


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # ---- simultaneous, N_SEEDS runs ----
    print(f"\nSimultaneous (3-plane), {N_SEEDS} seeds, {N_ITERS} iterations each...")
    sim_results = []  # each: [q_near, q_mid, q_far]
    for seed in range(N_SEEDS):
        _, hist, cps = multiplane_gs_torch(
            targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        final_iter = max(cps.keys())
        q = [masked_or_full_psnr(cps[final_iter][i], targets[i], i) for i in range(3)]
        rising = still_rising(hist)
        sim_results.append(q)
        print(f"  seed={seed}: near={q[0]:.2f} mid={q[1]:.2f} far={q[2]:.2f} dB"
              f"{' STILL RISING' if rising else ''}")

    # ---- sequential, N_SEEDS runs per plane ----
    print(f"\nSequential (single-plane), {N_SEEDS} seeds per plane, {N_ITERS} iterations each...")
    seq_results = {i: [] for i in range(3)}
    for i, name in enumerate(PLANE_NAMES):
        print(f"  Plane: {name}")
        for seed in range(N_SEEDS):
            _, hist, cps = multiplane_gs_torch(
                [targets[i]], [DEPTHS_M[i]], WAVELENGTH, DX, N_ITERS, device=DEVICE, seed=seed,
                pad_factor=PAD_FACTOR, smooth_cutoff=True
            )
            final_iter = max(cps.keys())
            q = masked_or_full_psnr(cps[final_iter][0], targets[i], i)
            rising = still_rising(hist)
            seq_results[i].append(q)
            print(f"    seed={seed}: {q:.2f} dB{' STILL RISING' if rising else ''}")

    # ---- summary ----
    print("\n" + "=" * 100)
    print(f"{'Plane':<8}{'Sim avg (range)':<28}{'Seq avg (range)':<28}{'Delta (avg)':<14}{'Delta range'}")
    all_deltas = {}
    for i, name in enumerate(PLANE_NAMES):
        sim_vals = [r[i] for r in sim_results]
        seq_vals = seq_results[i]
        sim_avg = sum(sim_vals) / len(sim_vals)
        seq_avg = sum(seq_vals) / len(seq_vals)
        sim_range = f"{min(sim_vals):.2f}-{max(sim_vals):.2f}"
        seq_range = f"{min(seq_vals):.2f}-{max(seq_vals):.2f}"
        delta_avg = seq_avg - sim_avg
        # all pairwise deltas across the 5x5 seed combinations, to see the full spread, not just avg-of-avg
        pairwise_deltas = [sv - sm for sv in seq_vals for sm in sim_vals]
        all_deltas[i] = pairwise_deltas
        print(f"{name:<8}{f'{sim_avg:.2f} ({sim_range})':<28}{f'{seq_avg:.2f} ({seq_range})':<28}"
              f"{delta_avg:<+14.2f}{min(pairwise_deltas):+.2f} to {max(pairwise_deltas):+.2f}")
    print("=" * 100)

    print(f"\nOriginal single-seed finding: near ~0dB, mid ~0dB, far +2.1 to +2.8dB (sequential advantage).")
    for i, name in enumerate(PLANE_NAMES):
        deltas = all_deltas[i]
        crosses_zero = min(deltas) < 0 < max(deltas)
        consistently_positive = min(deltas) > 0.3
        consistently_near_zero = max(abs(d) for d in deltas) < 1.0
        if name == "far":
            if consistently_positive:
                print(f"{name}: sequential advantage HOLDS across all seed pairings "
                      f"({min(deltas):+.2f} to {max(deltas):+.2f} dB) -- real, reproducible finding.")
            elif crosses_zero:
                print(f"{name}: delta crosses zero across seed pairings ({min(deltas):+.2f} to {max(deltas):+.2f} dB) --")
                print(f"    the advantage is NOT reliably reproducible; the original single-seed result may have")
                print(f"    been a lucky draw, not a real, consistent effect.")
            else:
                print(f"{name}: mixed signal, {min(deltas):+.2f} to {max(deltas):+.2f} dB -- weaker/less consistent")
                print(f"    than the original single-seed result suggested.")
        else:
            if consistently_near_zero:
                print(f"{name}: no meaningful delta either way ({min(deltas):+.2f} to {max(deltas):+.2f} dB) -- "
                      f"confirms the original 'no advantage' finding.")
            else:
                print(f"{name}: delta range is wider than expected ({min(deltas):+.2f} to {max(deltas):+.2f} dB) -- "
                      f"the original 'no advantage' claim may not be as tight as one seed suggested.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, name in enumerate(PLANE_NAMES):
        sim_vals = [r[i] for r in sim_results]
        seq_vals = seq_results[i]
        axes[i].scatter([0] * len(sim_vals), sim_vals, label="Simultaneous", color="tab:blue")
        axes[i].scatter([1] * len(seq_vals), seq_vals, label="Sequential", color="tab:green")
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Simultaneous", "Sequential"])
        axes[i].set_title(name)
        axes[i].set_ylabel("PSNR (dB)")
        if i == 0:
            axes[i].legend()
    plt.tight_layout()
    plt.savefig("seq_vs_sim_seed_check.png", dpi=130)
    print("Saved plot: seq_vs_sim_seed_check.png")


if __name__ == "__main__":
    main()
