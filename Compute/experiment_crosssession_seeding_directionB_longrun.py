"""
Follow-up to experiment_crosssession_seeding_v3.py, Direction B only
(synthetic donor -> realistic target).

That run found donor-seeding reaching a slightly LOWER final quality
than random init (11.63 vs 11.83 dB, -0.2dB) at an 80-iteration budget
-- but neither condition's convergence curve looked fully flat at
iteration 80 in the saved plot, blue (random) especially. That -0.2dB
gap is either real and permanent, or an artifact of random simply not
having finished converging yet within the 80-iteration window while
donor-seeding (starting from a head start) had already quietly settled.
Those are different conclusions and this distinguishes them directly by
rerunning with N_ITERS_FULL=200 (2.5x the previous budget) for both
conditions, checking find_plateau's still_rising flag explicitly for
both, and comparing final quality only once both are actually flat.

Same 6 donors / 6 baseline seeds as v3's Direction B, same donor
solve budget (80 iters on synthetic content -- those already plateaued
well within that budget, no change needed there).

Run on your 3060 (expect ~20+ minutes):
    python3 experiment_crosssession_seeding_directionB_longrun.py
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
N_ITERS_FULL = 200             # up from 80 in v3 -- specifically to let both conditions actually flatten
DONOR_SEEDS = [0, 1, 2, 3, 4, 5]
BASELINE_SEEDS = [10, 11, 12, 13, 14, 15]


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()

    targets_realistic = make_realistic_multiplane_target(SHAPE)
    targets_synthetic = make_multiplane_target(SHAPE, soft=True)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets_realistic, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print(f"\nSolving {len(DONOR_SEEDS)} donor phases on synthetic content ({N_ITERS_DONOR} iters each)...")
    donors = []
    for seed in DONOR_SEEDS:
        donor_phase, donor_hist, _ = multiplane_gs_torch(
            targets_synthetic, DEPTHS_M, WAVELENGTH, DX, N_ITERS_DONOR, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        d_plateau, d_final, d_rising = find_plateau(donor_hist)
        donors.append(donor_phase)
        print(f"  donor seed={seed}: plateau@{d_plateau}/{N_ITERS_DONOR}, final {d_final:.2f} dB"
              f"{' STILL RISING' if d_rising else ''}")

    print(f"\nRandom-init baseline on the realistic target, {len(BASELINE_SEEDS)} seeds, "
          f"{N_ITERS_FULL} iterations...")
    baseline_results = []
    for seed in BASELINE_SEEDS:
        _, history, _ = multiplane_gs_torch(
            targets_realistic, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        plateau, final, rising = find_plateau(history)
        baseline_results.append({"seed": seed, "plateau": plateau, "final": final,
                                  "rising": rising, "history": history})
        print(f"  seed={seed}: plateau@{plateau}/{N_ITERS_FULL}, final {final:.2f} dB"
              f"{' STILL RISING' if rising else ''}")

    print(f"\nDonor-seeded runs on the realistic target, {N_ITERS_FULL} iterations...")
    donor_results = []
    for seed, donor_phase in zip(DONOR_SEEDS, donors):
        _, history, _ = multiplane_gs_torch(
            targets_realistic, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE,
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
    print(f"DIRECTION B, LONG RUN ({N_ITERS_FULL} iterations)")
    print(f"{'Condition':<20}{'Plateau iters':<32}{'Avg':<10}{'Final dB avg'}")
    print(f"{'Random init':<20}{str(baseline_plateaus):<32}{baseline_avg:<10.1f}{baseline_final_avg:.2f}")
    print(f"{'Donor-seeded':<20}{str(donor_plateaus):<32}{donor_avg:<10.1f}{donor_final_avg:.2f}")
    print("=" * 100)
    print(f"Random still rising at iter {N_ITERS_FULL}: seeds {baseline_still_rising if baseline_still_rising else 'none'}")
    print(f"Donor-seeded still rising at iter {N_ITERS_FULL}: seeds {donor_still_rising if donor_still_rising else 'none'}")

    quality_gap = donor_final_avg - baseline_final_avg
    print(f"\n80-iteration quality gap was -0.20 dB (donor below random).")
    print(f"{N_ITERS_FULL}-iteration quality gap: {quality_gap:+.2f} dB")

    if baseline_still_rising and not donor_still_rising:
        print("\nConfirmed: random was still climbing at 80 iterations while donor-seeded had already")
        print("settled -- the 80-iter comparison was unfair to random, not a real donor-seeding cost.")
    elif abs(quality_gap) < 0.05:
        print("\nGap has closed at true convergence -- the -0.2dB difference was a convergence-budget")
        print("artifact, not a real property of donor-seeding. Both reach the same final quality; only")
        print("the PATH there differs (donor-seeded gets there in fewer iterations).")
    elif quality_gap < -0.1:
        print("\nGap persists (or widened) even once both conditions are flat -- this IS a real, small")
        print("quality cost from donor-seeding, not a convergence-budget artifact. Worth noting alongside")
        print("the iteration-count win, not just the win alone.")
    else:
        print("\nGap shrank but didn't fully close -- partial artifact, partial real effect. Report both")
        print("numbers rather than picking one.")

    speedup_pct = (1 - donor_avg / baseline_avg) * 100
    print(f"\nIteration-count advantage at {N_ITERS_FULL}-iteration budget: {speedup_pct:.0f}% "
          f"(vs. 64% at the 80-iteration budget) -- check whether the speedup itself also shrinks")
    print("once random is given enough room to catch up on iteration count too, not just quality.")

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
    ax.set_title(f"Direction B long run: does the -0.2dB gap survive true convergence?\n"
                 f"({N_ITERS_FULL} iterations, realistic target, full resolution)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("crosssession_directionB_longrun.png", dpi=130)
    print("Saved plot: crosssession_directionB_longrun.png")


if __name__ == "__main__":
    main()
