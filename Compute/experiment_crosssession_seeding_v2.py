"""
Larger-sample, determinism-checked follow-up to
experiment_crosssession_seeding.py.

That first run (3 donors, 3 baseline seeds) found donor-seeding (reusing
a converged phase from UNRELATED content as the init for a genuinely
different target) beat random init on average (24 vs 33 iterations to
plateau) with less spread and no quality cost -- but on a sample too
small to trust given how much seed count has mattered everywhere else
in this project. This reruns with 5 donors and 5 fresh baseline seeds
(matching this project's usual n=5 bar, per 10.1/Appendix A.7), and
explicitly verifies determinism the way experiment_smart_init.py did for
its deterministic initializers: since multiplane_gs_torch only uses its
`seed` argument when init_phase is None (see retrieval_torch.py), a
donor-seeded run should be BIT-IDENTICAL regardless of what `seed` value
gets passed alongside it. One donor's run is repeated with a different
nominal seed to confirm this directly rather than assume it.

Run on your 3060 (expect ~10 minutes):
    python3 experiment_crosssession_seeding_v2.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
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
N_ITERS_FULL = 80              # trimmed from 100 (v1) to keep the larger sample's runtime reasonable --
                                # every full-res plateau seen so far in this project is well under 80
DONOR_SEEDS = [0, 1, 2, 3, 4]
BASELINE_SEEDS = [10, 11, 12, 13, 14]


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()

    targets_donor = make_realistic_multiplane_target(SHAPE)
    targets_test = make_multiplane_target(SHAPE, soft=True)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets_test, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # ---- donors ----
    print(f"\nSolving {len(DONOR_SEEDS)} donor phases on realistic content ({N_ITERS_DONOR} iters each)...")
    donors = []
    for seed in DONOR_SEEDS:
        donor_phase, donor_hist, _ = multiplane_gs_torch(
            targets_donor, DEPTHS_M, WAVELENGTH, DX, N_ITERS_DONOR, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        d_plateau, d_final, d_rising = find_plateau(donor_hist)
        donors.append(donor_phase)
        print(f"  donor seed={seed}: plateau@{d_plateau}/{N_ITERS_DONOR}, final {d_final:.2f} dB"
              f"{' STILL RISING' if d_rising else ''}")

    # ---- baseline ----
    print(f"\nEstablishing random-init baseline on the synthetic target, {len(BASELINE_SEEDS)} seeds...")
    baseline_results = []
    for seed in BASELINE_SEEDS:
        _, history, _ = multiplane_gs_torch(
            targets_test, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        plateau, final, rising = find_plateau(history)
        baseline_results.append({"seed": seed, "plateau": plateau, "final": final, "history": history})
        print(f"  seed={seed}: plateau@{plateau}/{N_ITERS_FULL}, final {final:.2f} dB"
              f"{' STILL RISING' if rising else ''}")

    # ---- donor-seeded runs ----
    print(f"\nSeeding the synthetic target from each donor phase...")
    donor_results = []
    for i, (seed, donor_phase) in enumerate(zip(DONOR_SEEDS, donors)):
        _, history, _ = multiplane_gs_torch(
            targets_test, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE,
            pad_factor=PAD_FACTOR, smooth_cutoff=True, init_phase=donor_phase
        )
        plateau, final, rising = find_plateau(history)
        donor_results.append({"donor_seed": seed, "plateau": plateau, "final": final, "history": history})
        print(f"  donor seed={seed}: plateau@{plateau}/{N_ITERS_FULL}, final {final:.2f} dB"
              f"{' STILL RISING' if rising else ''}")

    # ---- determinism check: rerun donor 0's seeding with a DIFFERENT nominal `seed` ----
    print(f"\nDeterminism check: rerunning donor seed={DONOR_SEEDS[0]}'s seeded solve with a different "
          f"nominal `seed` argument (should be bit-identical, since init_phase makes `seed` unused)...")
    _, history_repeat, _ = multiplane_gs_torch(
        targets_test, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE, seed=9999,
        pad_factor=PAD_FACTOR, smooth_cutoff=True, init_phase=donors[0]
    )
    is_identical = history_repeat == donor_results[0]["history"]
    print(f"  determinism check: {'PASS -- bit-identical' if is_identical else 'FAIL -- NOT deterministic, investigate'}")

    baseline_plateaus = [r["plateau"] for r in baseline_results]
    baseline_min, baseline_max = min(baseline_plateaus), max(baseline_plateaus)
    baseline_avg = sum(baseline_plateaus) / len(baseline_plateaus)
    baseline_final_avg = sum(r["final"] for r in baseline_results) / len(baseline_results)

    donor_plateaus = [r["plateau"] for r in donor_results]
    donor_min, donor_max = min(donor_plateaus), max(donor_plateaus)
    donor_avg = sum(donor_plateaus) / len(donor_plateaus)
    donor_final_avg = sum(r["final"] for r in donor_results) / len(donor_results)

    print("\n" + "=" * 90)
    print(f"{'Condition':<30}{'Plateau iters':<30}{'Avg plateau':<14}{'Final dB'}")
    print(f"{'Random init (5 seeds)':<30}{str(baseline_plateaus):<30}{baseline_avg:<14.1f}{baseline_final_avg:.2f}")
    print(f"{'Donor-seeded (5 donors)':<30}{str(donor_plateaus):<30}{donor_avg:<14.1f}{donor_final_avg:.2f}")
    print("=" * 90)
    print(f"Determinism check: {'PASS' if is_identical else 'FAIL'}")

    if donor_max <= baseline_min:
        print("\nEvery donor beats random's fastest seed -- strong, general-purpose effect. Worth escalating")
        print("into the compute estimate as a real, low-cost (one-time, amortized) initialization strategy.")
    elif donor_avg < baseline_avg:
        pct = (1 - donor_avg / baseline_avg) * 100
        print(f"\nDonor-seeding beats the random average by {pct:.0f}% at n=5 each -- holds up from the n=3 result.")
        print("Consistent with a real, general-purpose 'any real phase beats random noise' effect, not a")
        print("small-sample fluke. Combined with the one-time/amortized cost structure (unlike low-res")
        print("seeding), this is the strongest initialization candidate found in this experiment family.")
    else:
        print("\nDoes not beat random init on average at n=5 -- the n=3 result does not replicate at a larger")
        print("sample. Treat the earlier positive result as noise, not a real effect.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(baseline_results):
        ax.plot(range(1, N_ITERS_FULL + 1), r["history"], color="tab:blue", alpha=0.5,
                 label="Random init" if i == 0 else None)
    for i, r in enumerate(donor_results):
        ax.plot(range(1, N_ITERS_FULL + 1), r["history"], color="tab:green", alpha=0.5,
                 label="Donor-seeded" if i == 0 else None)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"Cross-session seeding, n=5 each, full resolution ({SHAPE[1]}x{SHAPE[0]})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("crosssession_seeding_v2_comparison.png", dpi=130)
    print("Saved plot: crosssession_seeding_v2_comparison.png")


if __name__ == "__main__":
    main()
