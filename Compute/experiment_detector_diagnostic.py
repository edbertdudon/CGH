"""
Diagnostic for the padding-reconciliation anomaly: at pad_factor=2.0,
300-iteration budget, find_plateau_robust reported plateau=220 for
seed=0 (vs. the official baseline's ~30-65 iteration range using the
older first-touch detector over only 100 iterations) -- yet still_rising
was False for every seed, meaning the curve was NOT meaningfully still
climbing by the end of the run.

That combination (plateau detected very late, but not "still rising")
suggests one of two things:
  A) Quality genuinely keeps drifting/settling in small increments out
     to iteration ~200+, just too slowly to trip the still_rising check
     (>0.2dB over the last 20 iterations) -- a real, previously unknown
     slow-tail effect.
  B) Quality reaches something close to final by iteration ~30-65 (as
     the original detector found), but has occasional LATE, isolated
     noise wobbles exceeding the 0.2dB tolerance -- and find_plateau_robust's
     "every remaining iteration must stay within tolerance" requirement
     means a single late wobble, however isolated, pushes the detected
     plateau far later than where the curve actually settled. The longer
     the budget, the more chances for one chance wobble to occur --
     a previously untested failure mode of this detector at long budgets
     (A.7 only validated it against 100-iteration histories).

This reruns pad_factor=2.0 at seeds 0, 3, 4 (the ones that showed late
plateaus in the reconciliation run) for 300 iterations, saves the full
per-iteration history, and reports exactly where in the run each
tolerance violation occurs -- distinguishing "still genuinely improving"
from "one late random wobble in an otherwise-flat tail".

Run on your 3060 (expect ~7 min):
    python3 experiment_detector_diagnostic.py
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch
from convergence import find_plateau_robust, find_plateau

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 300
PAD_FACTOR = 2.0
SEEDS = [0, 3, 4]


def main():
    print(f"Device: {DEVICE}")
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    histories = {}
    for seed in SEEDS:
        print(f"\nseed={seed}, running {N_ITERS} iterations...")
        _, history, _ = multiplane_gs_torch(
            targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        histories[seed] = history

        plateau_robust, reference, still_rising = find_plateau_robust(history)
        plateau_orig, final_orig, still_rising_orig = find_plateau(history, tol=0.2)

        print(f"   Original (first-touch, tol=0.2) detector: plateau={plateau_orig}, "
              f"still_rising={still_rising_orig}")
        print(f"   Robust (sustained-hold, tol=0.2, hold=10) detector: plateau={plateau_robust}, "
              f"reference={reference:.2f} dB, still_rising={still_rising}")
        print(f"   Quality at iter 65: {history[64]:.2f} dB | iter {plateau_robust}: "
              f"{history[plateau_robust-1]:.2f} dB | final (iter {N_ITERS}): {history[-1]:.2f} dB")

        # find every point from 65 onward that violates the robust detector's tolerance
        # band around the reference value -- i.e. every "wobble" that could have pushed
        # the detected plateau later
        violations = [(i + 1, v) for i, v in enumerate(history) if i >= 64 and abs(v - reference) >= 0.2]
        if violations:
            last_violation_iter = violations[-1][0]
            print(f"   Violations of |quality - {reference:.2f}| >= 0.2dB after iter 65: "
                  f"{len(violations)} points, last at iter {last_violation_iter}")
            print(f"   First few violations: {violations[:5]}")
            print(f"   Last few violations: {violations[-5:]}")
        else:
            print("   No violations after iter 65 -- plateau=65 or earlier should have been found; "
                  "unexpected if plateau_robust is much later than 65.")

        net_change_65_to_final = history[-1] - history[64]
        print(f"   Net quality change from iter 65 to final: {net_change_65_to_final:+.3f} dB "
              f"({'genuine late improvement' if abs(net_change_65_to_final) > 0.3 else 'flat -- consistent with isolated late noise, not real drift'})")

    fig, ax = plt.subplots(figsize=(10, 6))
    for seed in SEEDS:
        ax.plot(range(1, N_ITERS + 1), histories[seed], label=f"seed={seed}", linewidth=1)
    ax.axvline(65, color="gray", linestyle=":", label="iter 65 (original detector's typical range)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Quality (internal amplitude metric, dB)")
    ax.set_title(f"Full convergence histories, pad_factor={PAD_FACTOR}, {N_ITERS} iterations")
    ax.legend()
    plt.tight_layout()
    plt.savefig("detector_diagnostic.png", dpi=130)
    print("\nSaved plot: detector_diagnostic.png")


if __name__ == "__main__":
    main()
