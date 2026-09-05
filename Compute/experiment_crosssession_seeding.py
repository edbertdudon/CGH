"""
Third and structurally distinct initialization test, after
experiment_smart_init.py (flat backprop guess -- lost) and
experiment_lowres_seeding.py (synthesized coarse-to-fine guess -- lost
on total FFT cost both times tried). Those both tried to SYNTHESIZE an
informed guess directly from the target being solved. This tests a
different hypothesis entirely: does reusing a REAL, fully-converged
phase -- solved for completely unrelated content -- work as a general-
purpose default initializer, better than random noise, purely because it
already satisfies "phase-only, unit amplitude" and carries whatever
generic structure real converged CGH phases tend to share (vs. i.i.d.
random noise, which carries none)?

This reuses 10.5's warm-start mechanism (init_phase=), but outside the
frame-to-frame context it was built for and already validated in. There,
consecutive frames share real content similarity, which is why it
worked. Here there is no assumed content relationship at all -- the
donor phase is solved for one content type (the realistic near/mid/far
scene used throughout this project) and reused, unmodified, to seed a
genuinely different content type (synthetic disc/ring/checker, the same
"scene cut" content used in the temporal warm-start family's negative-
transfer test). If this helps here, it's a general "borrowed real
solution beats random" effect, independent of content similarity --
notably the OPPOSITE of what the scene-cut test found (warm-starting
across a content mismatch was actively harmful at small scale). If it
doesn't help, that result is consistent with the scene-cut finding
rather than contradicting it.

Three donor phases (solved from the realistic-content target, 3 random
seeds, run to their own plateau) are each tried as the init for the
SAME synthetic target, and compared against a freshly-established
random-init baseline for that synthetic target (never measured before
in this project at full resolution -- all prior full-res baselines used
the realistic content).

Run on your 3060 (expect several minutes):
    python3 experiment_crosssession_seeding.py
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
N_ITERS_DONOR = 80             # generous -- donor phases from experiment_end_to_end_seed_check.py's own
                                # data plateaued by iter 65 at worst; 80 ensures each donor is well-converged
N_ITERS_FULL = 100             # matches every other full-res convergence sweep in this project
DONOR_SEEDS = [0, 1, 2]
BASELINE_SEEDS = [10, 11, 12]  # fresh seeds for the synthetic-target baseline, disjoint from donor seeds


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()

    targets_donor = make_realistic_multiplane_target(SHAPE)      # content the donor phases are solved for
    targets_test = make_multiplane_target(SHAPE, soft=True)      # unrelated content the donors get reused on

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets_test, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # ---- step 1: solve donor phases on the REALISTIC content, unrelated to the test target ----
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
              f"{' STILL RISING -- donor may not be fully converged' if d_rising else ''}")

    # ---- step 2: fresh random-init baseline on the SYNTHETIC test target (never measured before) ----
    print(f"\nEstablishing random-init baseline on the SYNTHETIC target, {len(BASELINE_SEEDS)} seeds...")
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

    # ---- step 3: seed the SAME synthetic target from each donor phase ----
    print(f"\nSeeding the synthetic target from each donor phase...")
    donor_results = []
    for i, (seed, donor_phase) in enumerate(zip(DONOR_SEEDS, donors)):
        counter.reset()
        _, history, _ = multiplane_gs_torch(
            targets_test, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE,
            pad_factor=PAD_FACTOR, smooth_cutoff=True, init_phase=donor_phase
        )
        plateau, final, rising = find_plateau(history)
        donor_results.append({"donor_seed": seed, "plateau": plateau, "final": final, "history": history})
        print(f"  donor seed={seed}: plateau@{plateau}/{N_ITERS_FULL}, final {final:.2f} dB"
              f"{' STILL RISING' if rising else ''}")

    baseline_plateaus = [r["plateau"] for r in baseline_results]
    baseline_min, baseline_max = min(baseline_plateaus), max(baseline_plateaus)
    baseline_avg = sum(baseline_plateaus) / len(baseline_plateaus)

    donor_plateaus = [r["plateau"] for r in donor_results]
    donor_min, donor_max = min(donor_plateaus), max(donor_plateaus)
    donor_avg = sum(donor_plateaus) / len(donor_plateaus)

    print("\n" + "=" * 80)
    print(f"{'Condition':<28}{'Plateau iter':<20}{'Final dB'}")
    print(f"{'Random init (3 seeds)':<28}{f'{baseline_min}-{baseline_max} (avg {baseline_avg:.0f})':<20}"
          f"{sum(r['final'] for r in baseline_results)/len(baseline_results):.2f}")
    print(f"{'Donor-seeded (3 donors)':<28}{f'{donor_min}-{donor_max} (avg {donor_avg:.0f})':<20}"
          f"{sum(r['final'] for r in donor_results)/len(donor_results):.2f}")
    print("=" * 80)

    if donor_max <= baseline_min:
        print("\nEvery donor beats random's fastest seed -- a real, general-purpose init effect,")
        print("independent of content similarity. Worth folding into the compute estimate directly.")
    elif donor_avg < baseline_avg:
        print("\nDonor-seeding beats the random average -- some real benefit from reusing ANY converged")
        print("phase, even for unrelated content. Does not contradict the scene-cut finding (that tested")
        print("mid-sequence disruption, a different question) but does suggest 'any real phase beats")
        print("random noise' as a general default worth considering.")
    else:
        print("\nDoes not beat random init on average -- a borrowed converged phase from unrelated content")
        print("is not a generically useful starting point. Consistent with the scene-cut finding: phase")
        print("structure appears to be content-specific, not a generic property random noise lacks.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in baseline_results:
        ax.plot(range(1, N_ITERS_FULL + 1), r["history"], color="tab:blue", alpha=0.6,
                 label="Random init" if r is baseline_results[0] else None)
    for r in donor_results:
        ax.plot(range(1, N_ITERS_FULL + 1), r["history"], color="tab:green", alpha=0.6,
                 label="Donor-seeded" if r is donor_results[0] else None)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title("Cross-session seeding: donor phase (unrelated content) vs. random init\n"
                 "(synthetic target, full resolution)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("crosssession_seeding_comparison.png", dpi=130)
    print("Saved plot: crosssession_seeding_comparison.png")


if __name__ == "__main__":
    main()
