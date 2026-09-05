"""
Follow-up to experiment_smart_init.py. That test found a flat-phase
backprop initialization does NOT beat random's average, let alone its
fastest seed -- a single backward propagate with no phase information
just isn't informed enough to help. This tries a fundamentally different
kind of "informed" init: actually SOLVE the problem first, just cheaply.

Coarse-to-fine seeding: solve multi-plane GS at a heavily downsampled
resolution (same physical aperture, coarser sampling -- pixel pitch
scaled up so LOW_SHAPE * LOW_DX ~= SHAPE * DX, not just a smaller
aperture) against downsampled targets, for a modest iteration budget.
This is nearly free (a fraction of a second). Then upsample the
resulting phase to full resolution and use it as init_phase for the real
solve. This is standard multi-resolution/coarse-to-fine practice in
iterative optimization generally, not novel to CGH.

Upsampling detail that matters: phase wraps at +-pi, so naively
interpolating the raw phase VALUES creates false discontinuities at
wrap boundaries. Instead this upsamples the unit-magnitude COMPLEX FIELD
(real and imaginary parts separately, standard bilinear interpolation),
then takes the angle of the result -- avoids the wraparound artifact
entirely.

Tested directly against full resolution, matching this project's own
established baseline: 10.1/Appendix A.7 found random-phase init spans
30-65 plateau iterations across 5 seeds at this exact
resolution/content/algorithm (see experiment_end_to_end_seed_check.py).
Same bar as experiment_smart_init.py: not just "better than average" --
does it land near or below the FASTEST random seed, consistently, across
different low-res solve seeds (since the low-res stage itself is
seeded, this is not automatically zero-variance the way the flat
backprop guess was -- checked directly, not assumed).

Uses the ORIGINAL (non-robust) plateau detector from Demo/convergence.py
for direct comparability with the existing 30-65 baseline range, which
was itself measured with that detector.

Run on your 3060 (expect several minutes):
    python3 experiment_lowres_seeding.py
"""
import time
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch
from convergence import find_plateau

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)             # full target resolution, matches 10.1c / A.7
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_FULL = 100                # matches the 10.1c / A.7 baseline run exactly
DOWNSAMPLE_FACTOR = 10             # low-res solve at 1/10 linear resolution
LOW_SHAPE = (SHAPE[0] // DOWNSAMPLE_FACTOR, SHAPE[1] // DOWNSAMPLE_FACTOR)
LOW_DX = DX * DOWNSAMPLE_FACTOR    # keeps physical aperture size constant (coarser sampling, not a smaller aperture)
N_ITERS_LOWRES = 10                # cut from 30 after checking the low-res stage's own convergence:
                                    # it plateaus within 1-15 iterations depending on seed (see chat),
                                    # so the original 30-iteration budget was mostly wasted cost
LOWRES_SEEDS = [0, 1, 2]           # check whether the low-res stage's own seed still matters after upsampling

# established baseline, from experiment_end_to_end_seed_check.py -- same resolution/content/algorithm/detector
BASELINE_RANDOM_PLATEAUS = [36, 65, 62, 36, 30]


def downsample_target(arr, low_shape):
    t = torch.as_tensor(arr, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    down = F.interpolate(t, size=low_shape, mode="area")
    return down.squeeze(0).squeeze(0).cpu().numpy()


def upsample_phase(phase, target_shape):
    """Upsample via the unit complex field, not raw phase values, to avoid
    false discontinuities at the +-pi wrap boundary."""
    field = torch.exp(1j * phase)
    real = field.real.unsqueeze(0).unsqueeze(0)
    imag = field.imag.unsqueeze(0).unsqueeze(0)
    real_up = F.interpolate(real, size=target_shape, mode="bilinear", align_corners=False)
    imag_up = F.interpolate(imag, size=target_shape, mode="bilinear", align_corners=False)
    up_field = torch.complex(real_up.squeeze(0).squeeze(0), imag_up.squeeze(0).squeeze(0))
    return torch.angle(up_field)


def main():
    print(f"Device: {DEVICE}")
    print(f"Low-res solve: {LOW_SHAPE[1]}x{LOW_SHAPE[0]} at dx={LOW_DX*1e6:.1f}um "
          f"(same physical aperture as {SHAPE[1]}x{SHAPE[0]} at dx={DX*1e6:.1f}um)\n")

    t_start = time.time()
    targets_full = make_realistic_multiplane_target(SHAPE)
    targets_low = [downsample_target(t, LOW_SHAPE) for t in targets_full]

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets_full, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    results = []
    for seed in LOWRES_SEEDS:
        print(f"\n--- low-res seed={seed} ---")
        counter.reset()
        t0 = time.time()
        low_phase, low_history, _ = multiplane_gs_torch(
            targets_low, DEPTHS_M, WAVELENGTH, LOW_DX, N_ITERS_LOWRES,
            device=DEVICE, seed=seed, pad_factor=PAD_FACTOR
        )
        lowres_ffts = counter.count
        lowres_elapsed = time.time() - t0
        print(f"  low-res solve: {N_ITERS_LOWRES} iters, {lowres_ffts} FFTs, {lowres_elapsed:.2f}s, "
              f"final {low_history[-1]:.2f} dB (on downsampled target -- not comparable to full-res dB)")

        init_phase = upsample_phase(low_phase, SHAPE)

        counter.reset()
        t0 = time.time()
        full_phase, full_history, _ = multiplane_gs_torch(
            targets_full, DEPTHS_M, WAVELENGTH, DX, N_ITERS_FULL, device=DEVICE,
            pad_factor=PAD_FACTOR, smooth_cutoff=True, init_phase=init_phase
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        full_ffts = counter.count
        full_elapsed = time.time() - t0

        plateau_iter, final_psnr, still_rising = find_plateau(full_history)
        total_ffts_at_plateau = lowres_ffts + (full_ffts / N_ITERS_FULL) * plateau_iter

        results.append({
            "seed": seed, "plateau_iter": plateau_iter, "final": final_psnr,
            "lowres_ffts": lowres_ffts, "total_ffts_at_plateau": total_ffts_at_plateau,
            "history": full_history, "still_rising": still_rising,
        })
        print(f"  full-res (seeded from low-res): plateau@{plateau_iter}/{N_ITERS_FULL}, "
              f"final {final_psnr:.2f} dB, {full_elapsed:.1f}s"
              f"{' STILL RISING' if still_rising else ''}")
        print(f"  total FFTs at plateau (low-res solve + full-res to plateau): {total_ffts_at_plateau:.0f}")

    baseline_min = min(BASELINE_RANDOM_PLATEAUS)
    baseline_avg = sum(BASELINE_RANDOM_PLATEAUS) / len(BASELINE_RANDOM_PLATEAUS)
    baseline_max = max(BASELINE_RANDOM_PLATEAUS)

    print("\n" + "=" * 90)
    print(f"{'Low-res seed':<16}{'Full-res plateau':<20}{'Final dB':<12}{'Total FFTs@plateau':<20}")
    for r in results:
        print(f"{r['seed']:<16}{r['plateau_iter']:<20}{r['final']:<12.2f}{r['total_ffts_at_plateau']:<20.0f}")
    print("-" * 90)
    print(f"Reference -- random-init baseline (no low-res stage), 5 seeds: "
          f"{BASELINE_RANDOM_PLATEAUS} (min={baseline_min}, avg={baseline_avg:.0f}, max={baseline_max})")
    print("=" * 90)

    lr_plateaus = [r["plateau_iter"] for r in results]
    lr_min, lr_max = min(lr_plateaus), max(lr_plateaus)
    lr_avg = sum(lr_plateaus) / len(lr_plateaus)
    lr_spread = lr_max - lr_min

    print(f"\nLow-res-seeded plateau across {len(LOWRES_SEEDS)} low-res seeds: "
          f"{lr_plateaus} (min={lr_min}, avg={lr_avg:.0f}, max={lr_max}, spread={lr_spread})")

    if lr_max <= baseline_min:
        print("\nBeats or matches random's FASTEST seed on every low-res seed tried -- and the low-res")
        print("solve itself is cheap (near-zero FFT cost relative to full-res). This is the real thing:")
        print("worth escalating with more low-res seeds and folding into the compute estimate directly.")
    elif lr_avg < baseline_avg and lr_spread < (baseline_max - baseline_min):
        print("\nBeats the random average AND shows less seed-to-seed spread than raw random init --")
        print("a partial but real win: not a guaranteed fastest-possible solve, but more consistent and")
        print("on average faster. Worth a larger low-res-seed sweep before treating the range as settled.")
    elif lr_avg < baseline_avg:
        print("\nBeats the random average but spread is still comparable to raw random init -- some benefit,")
        print("doesn't solve the variance problem by itself.")
    else:
        print("\nDoes not beat the random average -- unlike the flat backprop guess, this DID actually solve")
        print("the problem at low res, so a negative result here is more informative: either the")
        print(f"downsample factor ({DOWNSAMPLE_FACTOR}x) is too aggressive to preserve useful structure, or")
        print("phase information doesn't transfer usefully across this resolution gap for this content.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        ax.plot(range(1, N_ITERS_FULL + 1), r["history"], label=f"low-res seed={r['seed']}", alpha=0.8)
    ax.set_xlabel("Full-resolution GS iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"Full-res convergence when seeded from a {LOW_SHAPE[1]}x{LOW_SHAPE[0]} solve\n"
                 f"(baseline random-init range: {baseline_min}-{baseline_max} iters to plateau)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("lowres_seeding_comparison.png", dpi=130)
    print("Saved plot: lowres_seeding_comparison.png")


if __name__ == "__main__":
    main()
