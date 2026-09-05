"""
Same long-budget retest as experiment_crosssession_seeding_directionB_longrun.py,
applied to Direction A (realistic donor -> synthetic target) instead of
Direction B.

Direction B's short-budget (80-iteration) "donor-seeding wins" result
completely reversed once both conditions were run to true convergence
(200 iterations, neither still rising): donor-seeding turned out to need
MORE iterations on average and land at slightly lower final quality --
the apparent early lead was a longer, more circuitous path, not a real
shortcut. Direction A's effect was smaller to begin with (24% speedup,
-0.03dB at 80 iterations) and has not been checked at true convergence
at all. This closes that gap using the identical methodology (same 6
donor seeds, same 6 baseline seeds, N_ITERS_FULL=200, explicit
still_rising check on both conditions) so the two directions are
directly comparable.

Run on your 3060 (expect ~20 minutes):
    python3 experiment_crosssession_seeding_directionA_longrun.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, make_multiplane_target
from retrieval_torch import multiplane_gs_torch
from convergence import find_plateau

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_DONOR = 80
N_ITERS_FULL = 200
DONOR_SEEDS = [0, 1, 2, 3, 4, 5]
BASELINE_SEEDS = [10, 11, 12, 13, 14, 15]


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()

    targets_realistic = make_realistic_multiplane_target(SHAPE)
    targets_synthetic = make_multiplane_target(SHAPE, soft=True)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets_synthetic, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print(f"\nSolving {len(DONOR_SEEDS)} donor phases on realistic content ({N_ITERS_DONOR} iters each)...")
    donors = []
    for seed in DONOR_SEEDS:
        donor_phase, donor_hist, _ = multiplane_gs_torch(
            targets_realistic, DEPTHS_M, WAVELENGTH, DX, N_ITERS_DONOR, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        d_plateau, d_final, d_rising = find_plateau(donor_hist)
        donors.append(donor_phase)
        print(f"  donor seed={seed}: plateau@{d_plateau}/{N_ITERS_DONOR}, final {d_final:.2f} dB"
              f"{' STILL RISING' if d_rising else ''}")

    print(f"\nRandom-init baseline on the synthetic target, {len(BASELINE_SEEDS)} seeds, "
          f"{N_ITERS_FULL} iterations...")
    baseline_results = []
    for seed in BASELINE_SEEDS:
        _, history, _ = multiplane_gs_torch(
            targets_synthetic, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        plateau, final, rising = find_plateau(history)
        baseline_results.append({"seed": seed, "plateau": plateau, "final": final,
                                  "rising": rising, "history": history})
        print(f"  seed={seed}: plateau@{plateau}/{N_ITERS_FULL}, final {final:.2f} dB"
              f"{' STILL RISING' if rising else ''}")

    print(f"\nDonor-seeded runs on the synthetic target, {N_ITERS_FULL} iterations...")
    donor_results = []
    for seed, donor_phase in zip(DONOR_SEEDS, donors):
        _, history, _ = multiplane_gs_torch(
            targets_synthetic, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE,
            pad_factor=PAD_FACTOR, smooth_cutoff=True, init_phase=donor_phase
        )
        plateau, final, rising = find_plateau(history)
        donor_results.append({"donor_seed": seed, "plateau": plateau, "final": final,
                               "rising": rising, "history": history})
        print(f"  donor seed={seed}: plateau@{plateau}/{N_ITERS_FULL}, final {final:.2f} dB"
              f"{' STILL RISING' if rising else ''}")

    baseline_plateaus = [r["plateau"] for r in baseline_results]
    donor_plateaus = [r["plateau"] for r in donor_results]
    baseline_avg = sum(baseline_plateaus) / len(baseline_plateaus)
    donor_avg = sum(donor_plateaus) / len(donor_plateaus)
    baseline_final_avg = sum(r["final"] for r in baseline_results) / len(baseline_results)
    donor_final_avg = sum(r["final"] for r in donor_results) / len(donor_results)
    baseline_still_rising = [r["seed"] for r in baseline_results if r["rising"]]
    donor_still_rising = [r["donor_seed"] for r in donor_results if r["rising"]]

    print("\n" + "=" * 100)
    print(f"DIRECTION A, LONG RUN ({N_ITERS_FULL} iterations)")
    print(f"{'Condition':<20}{'Plateau iters':<32}{'Avg':<10}{'Final dB avg'}")
    print(f"{'Random init':<20}{str(baseline_plateaus):<32}{baseline_avg:<10.1f}{baseline_final_avg:.2f}")
    print(f"{'Donor-seeded':<20}{str(donor_plateaus):<32}{donor_avg:<10.1f}{donor_final_avg:.2f}")
    print("=" * 100)
    print(f"Random still rising at iter {N_ITERS_FULL}: seeds {baseline_still_rising if baseline_still_rising else 'none'}")
    print(f"Donor-seeded still rising at iter {N_ITERS_FULL}: seeds {donor_still_rising if donor_still_rising else 'none'}")

    quality_gap = donor_final_avg - baseline_final_avg
    print(f"\n80-iteration quality gap was -0.03 dB (donor slightly below random, near noise level).")
    print(f"{N_ITERS_FULL}-iteration quality gap: {quality_gap:+.2f} dB")

    speedup_pct = (1 - donor_avg / baseline_avg) * 100
    print(f"Iteration-count advantage at {N_ITERS_FULL}-iteration budget: {speedup_pct:+.0f}% "
          f"(vs. +24% at the 80-iteration budget)")

    if speedup_pct > 10 and quality_gap > -0.1:
        print("\nEffect holds at true convergence -- unlike Direction B, this is a real, reproducible")
        print("advantage, not a short-budget artifact. Worth trusting for this content direction.")
    elif speedup_pct < 0:
        print("\nSame reversal as Direction B: donor-seeding needed MORE iterations once given room to")
        print("actually finish. The short-budget result was misleading in this direction too -- neither")
        print("direction of cross-session donor seeding survives a true-convergence check.")
    else:
        print("\nMixed: some shrinkage from the short-budget result but not a full reversal. Report the")
        print("weaker, true-convergence numbers, not the original short-budget claim.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(baseline_results):
        ax.plot(range(1, N_ITERS_FULL + 1), r["history"], color="tab:blue", alpha=0.5,
                 label="Random init" if i == 0 else None)
    for i, r in enumerate(donor_results):
        ax.plot(range(1, N_ITERS_FULL + 1), r["history"], color="tab:green", alpha=0.5,
                 label="Donor-seeded" if i == 0 else None)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"Direction A long run: does the earlier win survive true convergence?\n"
                 f"({N_ITERS_FULL} iterations, synthetic target, full resolution)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("crosssession_directionA_longrun.png", dpi=130)
    print("Saved plot: crosssession_directionA_longrun.png")


if __name__ == "__main__":
    main()
