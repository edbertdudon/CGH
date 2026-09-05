"""
Follow-up to experiment_int8_precision_fullres_converged.py.

That run found 8-bit's plateau at iteration 70 vs. float32's 37 -- a
number that would matter a lot if real (roughly halving the power win
from switching to INT8, since more iterations there means more total
FFTs even at higher TOPS/W) but was flagged as suspect: the plateau
detector (first iteration within 0.2dB of the run's own final value) is
known to be fragile near a flat, noisy convergence curve -- this
project's own off-by-one bug investigation found the exact same detector
swing wildly (58 vs 36 iterations) between two runs whose underlying
convergence data differed by only one shifted index.

This reruns float32 and 8-bit (the two conditions in question) across
several different random seeds, same content, same static quantization
scales (calibrated once, reused across all seeds and both conditions --
recalibrating per seed would conflate two different questions), same
full resolution and iteration budget as the run being checked. If the
~2x plateau gap replicates across seeds, it's real. If plateau_iter
swings wildly seed-to-seed for either condition, the detector's
fragility -- not a genuine INT8 convergence-rate difference -- is the
right explanation, and the single 70-vs-37 run shouldn't be trusted.

Run on your 3060 (expect several minutes):
    python3 experiment_int8_plateau_reproducibility.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from propagation_torch import angular_spectrum_propagate
from experiment_int8_precision_fullres_converged import (
    calibrate_scales, multiplane_gs_quantized_static, find_plateau,
    SHAPE, WAVELENGTH, DX, PAD_FACTOR, DEPTHS_M, N_ITERS_MAX, CALIBRATION_ITERS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
CONDITIONS = [("float32 (baseline)", None), ("8-bit", 8)]


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = angular_spectrum_propagate(
        torch.exp(1j * torch.rand(SHAPE, device=DEVICE)), WAVELENGTH, DX, DEPTHS_M[0], pad_factor=PAD_FACTOR
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print(f"Calibrating static scales once ({CALIBRATION_ITERS} iterations, seed=0) -- "
          f"shared across all seeds and both conditions below...")
    scales = calibrate_scales(targets, DEPTHS_M, WAVELENGTH, DX, CALIBRATION_ITERS,
                               device=DEVICE, pad_factor=PAD_FACTOR)
    print()

    results = {}
    for label, bits in CONDITIONS:
        results[label] = []
        for seed in SEEDS:
            print(f"Running {label}, seed={seed}...")
            t0 = time.time()
            history, per_plane = multiplane_gs_quantized_static(
                targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS_MAX, bits=bits, scales=scales,
                device=DEVICE, seed=seed, pad_factor=PAD_FACTOR
            )
            elapsed = time.time() - t0
            plateau_iter, final_psnr, still_rising = find_plateau(history)
            results[label].append({"seed": seed, "plateau_iter": plateau_iter, "final": final_psnr,
                                    "still_rising": still_rising, "history": history})
            print(f"   plateau@{plateau_iter}/{N_ITERS_MAX}, final {final_psnr:.2f} dB, {elapsed:.1f}s"
                  f"{' STILL RISING' if still_rising else ''}")

    print("\n" + "=" * 80)
    print(f"{'Condition':<20}{'Seed':<8}{'Plateau iter':<16}{'Final dB':<12}")
    for label, _ in CONDITIONS:
        for r in results[label]:
            print(f"{label:<20}{r['seed']:<8}{r['plateau_iter']:<16}{r['final']:<12.2f}")
    print("=" * 80)

    for label, _ in CONDITIONS:
        plateaus = [r["plateau_iter"] for r in results[label]]
        finals = [r["final"] for r in results[label]]
        spread = max(plateaus) - min(plateaus)
        print(f"\n{label}: plateau_iter across seeds = {plateaus} (spread: {spread}), "
              f"final dB spread = {max(finals) - min(finals):.2f}")

    fp32_plateaus = [r["plateau_iter"] for r in results["float32 (baseline)"]]
    int8_plateaus = [r["plateau_iter"] for r in results["8-bit"]]
    fp32_spread = max(fp32_plateaus) - min(fp32_plateaus)
    int8_spread = max(int8_plateaus) - min(int8_plateaus)
    gap_original = 70 - 37  # from the single-seed run being checked
    avg_gap = (sum(int8_plateaus) / len(int8_plateaus)) - (sum(fp32_plateaus) / len(fp32_plateaus))

    print(f"\nOriginal single-seed gap being checked: 8-bit plateau (70) - float32 plateau (37) = {gap_original}")
    print(f"Average gap across {len(SEEDS)} seeds here: {avg_gap:+.1f} iterations")
    print(f"Within-condition seed-to-seed spread: float32 {fp32_spread}, 8-bit {int8_spread}")

    if max(fp32_spread, int8_spread) >= abs(gap_original) * 0.5:
        print("\nWithin-condition spread is comparable to the original 70-vs-37 gap itself -- the detector's")
        print("known fragility, not a genuine INT8 convergence-rate difference, is the better explanation.")
        print("Do NOT treat 'INT8 needs ~2x more iterations' as established; final-quality parity (the")
        print("headline finding) is unaffected either way -- only the plateau-iteration side claim is in doubt.")
    else:
        print("\nSeed-to-seed spread within each condition is small relative to the gap between conditions --")
        print("the 8-bit-converges-slower finding looks real, not a detector artifact. Worth factoring into")
        print("the power-savings estimate: INT8's TOPS/W advantage is partly offset by needing more iterations.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, _ in CONDITIONS:
        for r in results[label]:
            ax.plot(range(1, len(r["history"]) + 1), r["history"],
                     label=f"{label}, seed={r['seed']}", alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title("Convergence curves across seeds: float32 vs. 8-bit (static scale, full res)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("int8_plateau_reproducibility.png", dpi=130)
    print("Saved plot: int8_plateau_reproducibility.png")


if __name__ == "__main__":
    main()
