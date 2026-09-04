"""
The end-to-end convergence sweep revealed every full-resolution
comparison this session (sequential vs. simultaneous, color native vs.
joint, real vs. synthetic content) used a fixed 16-iteration cutoff --
chosen because small-scale GS plateaus fast, which full-resolution GS
apparently does not. This re-checks the single highest-stakes
comparison -- sequential vs. simultaneous depth rendering, since it
decides whether 1,080Hz is needed at all -- at each condition's own
TRUE plateau instead of a shared fixed iteration count.

Both simultaneous and each of the three sequential single-plane solves
run up to N_ITERS_MAX iterations, with plateau auto-detected
independently for each (same logic as experiment_end_to_end_compute.py)
rather than assumed. This is an apples-to-apples true-convergence
comparison, not a snapshot.

If the ~0dB sequential advantage (icon/text) and ~+2.8dB advantage
(scene) both still hold at true convergence, that's real evidence the
other 16-iteration comparisons this session likely generalize too --
though only this one is actually being confirmed here. If either
result changes, the rest need re-checking as well.

This is a long run -- each of the 4 conditions may need up to
N_ITERS_MAX iterations before plateauing. Expect 10+ minutes.

Run on your 3060:
    python3 experiment_sequential_vs_simultaneous_converged.py
"""
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target, sparse_content_bounds
from retrieval_torch import multiplane_gs_torch
from metrics import psnr_intensity as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_MAX = 150
PLATEAU_TOLERANCE_DB = 0.2
PLANE_NAMES = ["near", "mid", "far"]
SPARSE_PLANES = {0, 1}


def masked_or_full_psnr(recon, target, plane_idx):
    if plane_idx in SPARSE_PLANES:
        rows, cols = sparse_content_bounds(target)
        return psnr_np(recon[rows, cols], target[rows, cols])
    return psnr_np(recon, target)


def find_plateau(history, tol=PLATEAU_TOLERANCE_DB):
    final = history[-1]
    plateau_iter = next(i + 1 for i, v in enumerate(history) if v >= final - tol)
    still_rising = len(history) >= 10 and (final - history[-10]) > tol
    return plateau_iter, final, still_rising


def run_and_time(fn, *args, **kwargs):
    counter.reset()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    result = fn(*args, **kwargs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return result, elapsed, counter.count


def main():
    print(f"Device: {DEVICE}")
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # ---- simultaneous, run to its own plateau ----
    print(f"\nSimultaneous: running up to {N_ITERS_MAX} iterations...")
    (phase_sim, hist_sim, cps_sim), t_sim, ffts_sim = run_and_time(
        multiplane_gs_torch, targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS_MAX,
        device=DEVICE, pad_factor=PAD_FACTOR, smooth_cutoff=True
    )
    final_sim_iter = max(cps_sim.keys())
    plateau_sim, final_sim_mean, rising_sim = find_plateau(hist_sim)
    q_sim = [masked_or_full_psnr(cps_sim[final_sim_iter][i], targets[i], i) for i in range(3)]
    print(f"  Plateau at iter {plateau_sim}/{N_ITERS_MAX} (mean {final_sim_mean:.2f} dB), "
          f"{'STILL RISING -- increase N_ITERS_MAX' if rising_sim else 'converged'}")
    print(f"  Per-plane quality at iter {final_sim_iter}: {[round(q,1) for q in q_sim]}")

    # ---- sequential: each plane run separately, to its own plateau ----
    seq_results = []
    for i, name in enumerate(PLANE_NAMES):
        print(f"\nSequential, {name}: running up to {N_ITERS_MAX} iterations...")
        (phase_i, hist_i, cps_i), t_i, ffts_i = run_and_time(
            multiplane_gs_torch, [targets[i]], [DEPTHS_M[i]], WAVELENGTH, DX, N_ITERS_MAX,
            device=DEVICE, seed=i, pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        final_i = max(cps_i.keys())
        plateau_i, final_i_mean, rising_i = find_plateau(hist_i)
        q_i = masked_or_full_psnr(cps_i[final_i][0], targets[i], i)
        print(f"  Plateau at iter {plateau_i}/{N_ITERS_MAX} ({final_i_mean:.2f} dB), "
              f"{'STILL RISING -- increase N_ITERS_MAX' if rising_i else 'converged'}")
        print(f"  Quality at iter {final_i}: {q_i:.2f} dB")
        seq_results.append({
            "quality": q_i, "plateau_iter": plateau_i, "history": hist_i,
            "recon": cps_i[final_i][0], "rising": rising_i
        })

    # ---- summary ----
    print("\n" + "=" * 70)
    print("TRUE-CONVERGENCE comparison (not a 16-iteration snapshot):")
    print(f"{'plane':<8}{'simultaneous':<16}{'sequential':<14}{'delta':<10}{'(16-iter delta, for ref)'}")
    ref_deltas_16iter = {"near": 1.4, "mid": 4.0, "far": 3.0}  # from earlier session results, whole-frame/masked mix
    for i, name in enumerate(PLANE_NAMES):
        delta = seq_results[i]["quality"] - q_sim[i]
        print(f"{name:<8}{q_sim[i]:<16.2f}{seq_results[i]['quality']:<14.2f}{delta:<+10.2f}"
              f"(~{ref_deltas_16iter[name]:+.1f} dB at 16 iters)")
    print("=" * 70)
    any_rising = rising_sim or any(r["rising"] for r in seq_results)
    if any_rising:
        print("WARNING: at least one condition was still rising at N_ITERS_MAX -- results above")
        print("are not yet fully converged. Increase N_ITERS_MAX and rerun before trusting this.")

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(hist_sim) + 1), hist_sim, label="Simultaneous", linewidth=2)
    for i, name in enumerate(PLANE_NAMES):
        ax.plot(range(1, len(seq_results[i]["history"]) + 1), seq_results[i]["history"],
                label=f"Sequential ({name})", linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("True-convergence comparison: simultaneous vs. sequential")
    ax.legend()
    plt.tight_layout()
    plt.savefig("seq_vs_sim_converged.png", dpi=130)
    print("Saved plot: seq_vs_sim_converged.png")

    fig2, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, name in enumerate(PLANE_NAMES):
        axes[0, i].imshow(cps_sim[final_sim_iter][i], cmap="gray")
        axes[0, i].set_title(f"Simultaneous, {name} ({q_sim[i]:.1f} dB)")
        axes[0, i].axis("off")
        axes[1, i].imshow(seq_results[i]["recon"], cmap="gray")
        axes[1, i].set_title(f"Sequential, {name} ({seq_results[i]['quality']:.1f} dB)")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("seq_vs_sim_converged_recon.png", dpi=130)
    print("Saved plot: seq_vs_sim_converged_recon.png")


if __name__ == "__main__":
    main()
