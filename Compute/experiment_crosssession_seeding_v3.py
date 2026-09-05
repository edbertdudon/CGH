"""
Third round of cross-session seeding, two extensions over v2:

  1. Larger sample: 6 donors / 6 baseline seeds (up from 5/5 in v2,
     3/3 in v1).
  2. BOTH directions of content transfer, not just one:
       Direction A (tested in v1/v2): donors solved on REALISTIC content,
         used to seed the SYNTHETIC target.
       Direction B (new): donors solved on SYNTHETIC content, used to
         seed the REALISTIC target.
     If only one direction shows the effect, that's evidence it's
     content-direction-specific (e.g. realistic-content phases carry
     more generically useful structure than synthetic-shape phases, or
     vice versa) rather than a general "any real converged phase beats
     random noise" property. If both directions show it, that's much
     stronger evidence for a general effect.

Same determinism check as v2 (one donor-seeded run repeated with a
different nominal `seed` argument, confirming init_phase makes `seed`
irrelevant) -- run once per direction.

Run on your 3060 (expect ~20 minutes -- this is the largest run in this
experiment family so far):
    python3 experiment_crosssession_seeding_v3.py
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
N_ITERS_FULL = 80
DONOR_SEEDS = [0, 1, 2, 3, 4, 5]
BASELINE_SEEDS = [10, 11, 12, 13, 14, 15]


def solve_donors(targets_donor_content, label):
    print(f"\nSolving {len(DONOR_SEEDS)} donor phases on {label} content ({N_ITERS_DONOR} iters each)...")
    donors = []
    for seed in DONOR_SEEDS:
        donor_phase, donor_hist, _ = multiplane_gs_torch(
            targets_donor_content, DEPTHS_M, WAVELENGTH, DX, N_ITERS_DONOR, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        d_plateau, d_final, d_rising = find_plateau(donor_hist)
        donors.append(donor_phase)
        print(f"  donor seed={seed}: plateau@{d_plateau}/{N_ITERS_DONOR}, final {d_final:.2f} dB"
              f"{' STILL RISING' if d_rising else ''}")
    return donors


def run_baseline(targets_test, label):
    print(f"\nEstablishing random-init baseline on the {label} target, {len(BASELINE_SEEDS)} seeds...")
    results = []
    for seed in BASELINE_SEEDS:
        _, history, _ = multiplane_gs_torch(
            targets_test, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        plateau, final, rising = find_plateau(history)
        results.append({"seed": seed, "plateau": plateau, "final": final, "history": history})
        print(f"  seed={seed}: plateau@{plateau}/{N_ITERS_FULL}, final {final:.2f} dB"
              f"{' STILL RISING' if rising else ''}")
    return results


def run_donor_seeded(targets_test, donors, label):
    print(f"\nSeeding the {label} target from each donor phase...")
    results = []
    for seed, donor_phase in zip(DONOR_SEEDS, donors):
        _, history, _ = multiplane_gs_torch(
            targets_test, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE,
            pad_factor=PAD_FACTOR, smooth_cutoff=True, init_phase=donor_phase
        )
        plateau, final, rising = find_plateau(history)
        results.append({"donor_seed": seed, "plateau": plateau, "final": final, "history": history})
        print(f"  donor seed={seed}: plateau@{plateau}/{N_ITERS_FULL}, final {final:.2f} dB"
              f"{' STILL RISING' if rising else ''}")
    return results


def check_determinism(targets_test, donor_phase, expected_history):
    _, history_repeat, _ = multiplane_gs_torch(
        targets_test, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE, seed=9999,
        pad_factor=PAD_FACTOR, smooth_cutoff=True, init_phase=donor_phase
    )
    return history_repeat == expected_history


def summarize(direction_label, baseline_results, donor_results, determinism_ok):
    baseline_plateaus = [r["plateau"] for r in baseline_results]
    donor_plateaus = [r["plateau"] for r in donor_results]
    baseline_avg = sum(baseline_plateaus) / len(baseline_plateaus)
    donor_avg = sum(donor_plateaus) / len(donor_plateaus)
    baseline_final_avg = sum(r["final"] for r in baseline_results) / len(baseline_results)
    donor_final_avg = sum(r["final"] for r in donor_results) / len(donor_results)

    print("\n" + "=" * 90)
    print(f"DIRECTION: {direction_label}")
    print(f"{'Condition':<30}{'Plateau iters':<30}{'Avg':<10}{'Final dB'}")
    print(f"{'Random init':<30}{str(baseline_plateaus):<30}{baseline_avg:<10.1f}{baseline_final_avg:.2f}")
    print(f"{'Donor-seeded':<30}{str(donor_plateaus):<30}{donor_avg:<10.1f}{donor_final_avg:.2f}")
    print(f"Determinism check: {'PASS' if determinism_ok else 'FAIL'}")
    print("=" * 90)

    if donor_avg < baseline_avg:
        pct = (1 - donor_avg / baseline_avg) * 100
        print(f"Donor-seeding beats random by {pct:.0f}% on average in this direction.")
    else:
        pct = (donor_avg / baseline_avg - 1) * 100
        print(f"Donor-seeding does NOT beat random in this direction ({pct:.0f}% worse on average).")

    return {"direction": direction_label, "baseline_avg": baseline_avg, "donor_avg": donor_avg,
            "baseline_plateaus": baseline_plateaus, "donor_plateaus": donor_plateaus}


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

    # ---- Direction A: realistic donors -> synthetic target ----
    donors_a = solve_donors(targets_realistic, "realistic")
    baseline_a = run_baseline(targets_synthetic, "synthetic")
    donor_results_a = run_donor_seeded(targets_synthetic, donors_a, "synthetic")
    det_a = check_determinism(targets_synthetic, donors_a[0], donor_results_a[0]["history"])
    summary_a = summarize("A: realistic donor -> synthetic target", baseline_a, donor_results_a, det_a)

    # ---- Direction B: synthetic donors -> realistic target ----
    donors_b = solve_donors(targets_synthetic, "synthetic")
    baseline_b = run_baseline(targets_realistic, "realistic")
    donor_results_b = run_donor_seeded(targets_realistic, donors_b, "realistic")
    det_b = check_determinism(targets_realistic, donors_b[0], donor_results_b[0]["history"])
    summary_b = summarize("B: synthetic donor -> realistic target", baseline_b, donor_results_b, det_b)

    print("\n" + "#" * 90)
    print("OVERALL")
    both_help = summary_a["donor_avg"] < summary_a["baseline_avg"] and summary_b["donor_avg"] < summary_b["baseline_avg"]
    if both_help:
        print("Effect holds in BOTH directions -- strong evidence for a general 'any real converged phase")
        print("beats random noise' property, not a one-off or content-direction-specific quirk.")
    else:
        print("Effect does NOT hold in both directions -- content-direction-specific, not fully general.")
        print("Report which direction(s) actually work, not a blanket claim.")
    print("#" * 90)

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for i, r in enumerate(baseline_a):
        axes[0].plot(range(1, N_ITERS_FULL + 1), r["history"], color="tab:blue", alpha=0.5,
                      label="Random init" if i == 0 else None)
    for i, r in enumerate(donor_results_a):
        axes[0].plot(range(1, N_ITERS_FULL + 1), r["history"], color="tab:green", alpha=0.5,
                      label="Donor-seeded" if i == 0 else None)
    axes[0].set_title("A: realistic donor -> synthetic target")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean PSNR (dB)")
    axes[0].legend()

    for i, r in enumerate(baseline_b):
        axes[1].plot(range(1, N_ITERS_FULL + 1), r["history"], color="tab:blue", alpha=0.5,
                      label="Random init" if i == 0 else None)
    for i, r in enumerate(donor_results_b):
        axes[1].plot(range(1, N_ITERS_FULL + 1), r["history"], color="tab:green", alpha=0.5,
                      label="Donor-seeded" if i == 0 else None)
    axes[1].set_title("B: synthetic donor -> realistic target")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("crosssession_seeding_v3_comparison.png", dpi=130)
    print("Saved plot: crosssession_seeding_v3_comparison.png")


if __name__ == "__main__":
    main()
