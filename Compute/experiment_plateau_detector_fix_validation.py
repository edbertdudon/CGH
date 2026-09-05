"""
Validates find_plateau_robust() (Demo/convergence.py) against the exact
instability that motivated it: experiment_end_to_end_seed_check.py found
the OLD detector's plateau iteration swinging 30-65 (spread 35) across 5
seeds on this same resolution/content/algorithm, despite final quality
being stable (dB spread under 0.6) across those same seeds.

Reruns those same 5 seeds once, and applies BOTH the old and new
detector to each resulting history -- so this is a direct, apples-to-
apples before/after comparison on identical underlying data, not a
separate run that could differ for unrelated reasons.

Run on your 3060 (expect a few minutes):
    python3 experiment_plateau_detector_fix_validation.py
"""
import math
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch
from convergence import find_plateau, find_plateau_robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 100
SEEDS = [0, 1, 2, 3, 4]

TOPS_PER_W = 6.5
POWER_BUDGET_MW = 500


def analytic_gflop_per_fft(n_pixels):
    return 5 * n_pixels * math.log2(n_pixels) / 1e9


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    gflop_per_fft = analytic_gflop_per_fft(SHAPE[0] * PAD_FACTOR * SHAPE[1] * PAD_FACTOR)

    old_results, new_results = [], []
    for seed in SEEDS:
        print(f"Running seed={seed}, {N_ITERS} iterations...")
        counter.reset()
        phase, history, cps = multiplane_gs_torch(
            targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, seed=seed,
            pad_factor=PAD_FACTOR, smooth_cutoff=True
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        ffts_per_iter = counter.count / N_ITERS

        old_plateau, old_final, old_rising = find_plateau(history)
        new_plateau, new_ref, new_rising = find_plateau_robust(history)

        def power_for(plateau_iter):
            ffts = ffts_per_iter * plateau_iter
            gflop = ffts * gflop_per_fft
            tflops = gflop * 360 / 1000
            return tflops / TOPS_PER_W * 1000, ffts

        old_power, old_ffts = power_for(old_plateau)
        new_power, new_ffts = power_for(new_plateau)

        old_results.append({"seed": seed, "plateau": old_plateau, "ffts": old_ffts, "power": old_power})
        new_results.append({"seed": seed, "plateau": new_plateau, "ffts": new_ffts, "power": new_power})
        print(f"   OLD: plateau@{old_plateau}, {old_ffts:.0f} FFTs, {old_power:.0f}mW"
              f"{' (still rising)' if old_rising else ''}")
        print(f"   NEW: plateau@{new_plateau}, {new_ffts:.0f} FFTs, {new_power:.0f}mW"
              f"{' (still rising)' if new_rising else ''}")

    print("\n" + "=" * 90)
    print(f"{'Seed':<8}{'OLD plateau':<14}{'OLD power(mW)':<16}{'NEW plateau':<14}{'NEW power(mW)':<16}")
    for o, n in zip(old_results, new_results):
        print(f"{o['seed']:<8}{o['plateau']:<14}{o['power']:<16.0f}{n['plateau']:<14}{n['power']:<16.0f}")
    print("=" * 90)

    old_plateaus = [r["plateau"] for r in old_results]
    new_plateaus = [r["plateau"] for r in new_results]
    old_powers = [r["power"] for r in old_results]
    new_powers = [r["power"] for r in new_results]

    old_spread = max(old_plateaus) - min(old_plateaus)
    new_spread = max(new_plateaus) - min(new_plateaus)
    old_power_spread_pct = (max(old_powers) - min(old_powers)) / (sum(old_powers) / len(old_powers)) * 100
    new_power_spread_pct = (max(new_powers) - min(new_powers)) / (sum(new_powers) / len(new_powers)) * 100

    print(f"\nOLD detector: plateau spread = {old_spread} iterations, power spread = {old_power_spread_pct:.0f}% of mean")
    print(f"NEW detector: plateau spread = {new_spread} iterations, power spread = {new_power_spread_pct:.0f}% of mean")
    print(f"\nMean power -- OLD: {sum(old_powers)/len(old_powers):.0f}mW, NEW: {sum(new_powers)/len(new_powers):.0f}mW")

    if new_spread < old_spread * 0.5:
        print("\nFix confirmed: sustained-hold + averaged-reference detector meaningfully reduces seed-to-seed")
        print("spread vs. the old first-touch/single-sample detector on the exact same underlying data.")
    else:
        print("\nFix did NOT meaningfully reduce spread on this data -- the sustained-hold criterion alone may")
        print("not be enough; consider a larger hold window or a slope-based criterion instead.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
