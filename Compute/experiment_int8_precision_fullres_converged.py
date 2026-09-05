"""
Full-resolution, true-convergence retest of experiment_int8_precision.py
/ _static_scale.py.

Both prior INT8 tests ran at 512x512 with a fixed 16-iteration budget --
useful for a fast relative signal, but this project's own history (10.3's
sequential-rendering advantage, real at 512x512, gone at 4,700x2,700) is
the standing reason not to trust a small-scale finding until it's been
checked at the real target resolution and run to actual convergence
(10.1c found full-res GS needs far more than 16 iterations to plateau).

Uses the STATIC-scale quantization model (calibrated once from the
unquantized baseline, not oracle-rescaled per call) validated in
_static_scale.py as the more realistic emulation of a real fixed-point
datapath. Same insertion points (after each forward propagate, after the
amplitude constraint, after each backward propagate, on the accumulated
SLM field).

Bit depths reduced to [4, 8, 16, None] (skip the intermediate 6/10/12
points from the small-scale sweep) to keep full-resolution runtime
reasonable -- this is a scale check on the two data points that matter
most (INT8-equivalent, and the most aggressive case that showed real
degradation at small scale), not a full replication of the sweep.

Run on your 3060 (expect several minutes):
    python3 experiment_int8_precision_fullres_converged.py
"""
import math
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from propagation_torch import angular_spectrum_propagate, safe_abs
from retrieval_torch import _to_tensor_targets, _psnr_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)          # doc target resolution, matches 10.1c
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_MAX = 100              # matches 10.1c's convergence sweep budget
PLATEAU_TOLERANCE_DB = 0.2     # matches 10.1c / experiment_sequential_vs_simultaneous_converged.py
CALIBRATION_ITERS = 60         # enough to see representative dynamic range without a full 100-iter run
BITS_SWEEP = [4, 8, 16, None]  # reduced from the small-scale sweep's 7 points -- see docstring
PLANE_NAMES = ["near", "mid", "far"]
SLOTS = ["field_at_plane", "constrained", "back", "slm_field"]


def tensor_max(field):
    return torch.maximum(field.real.abs().max(), field.imag.abs().max()).clamp(min=1e-12)


def quantize_static(field, bits, fixed_scale):
    if bits is None:
        return field
    levels = 2 ** (bits - 1) - 1
    scale = fixed_scale / levels
    real_q = torch.round(field.real / scale) * scale
    imag_q = torch.round(field.imag / scale) * scale
    return torch.complex(real_q, imag_q)


def calibrate_scales(target_planes, depths_m, wavelength, dx, n_iters, device="cuda", seed=0, pad_factor=2.0):
    shape = target_planes[0].shape
    targets_amp = _to_tensor_targets(target_planes, device)

    torch.manual_seed(seed)
    slm_phase = (torch.rand(shape, device=device, dtype=torch.float32) * 2 * math.pi - math.pi)
    slm_field = torch.exp(1j * slm_phase)

    worst = {slot: torch.tensor(0.0, device=device) for slot in SLOTS}
    worst["slm_field"] = tensor_max(slm_field)

    for it in range(1, n_iters + 1):
        correction = torch.zeros(shape, dtype=torch.complex64, device=device)
        for target_amp, z in zip(targets_amp, depths_m):
            field_at_plane = angular_spectrum_propagate(slm_field, wavelength, dx, z, pad_factor=pad_factor)
            worst["field_at_plane"] = torch.maximum(worst["field_at_plane"], tensor_max(field_at_plane))

            phase_at_plane = torch.angle(field_at_plane)
            constrained = target_amp.to(torch.complex64) * torch.exp(1j * phase_at_plane.to(torch.float32))
            worst["constrained"] = torch.maximum(worst["constrained"], tensor_max(constrained))

            back = angular_spectrum_propagate(constrained, wavelength, dx, -z, pad_factor=pad_factor)
            worst["back"] = torch.maximum(worst["back"], tensor_max(back))
            correction += back

        slm_field = correction / len(depths_m)
        slm_phase = torch.angle(slm_field)
        slm_field = torch.exp(1j * slm_phase)
        worst["slm_field"] = torch.maximum(worst["slm_field"], tensor_max(slm_field))

    return {k: float(v) for k, v in worst.items()}


def multiplane_gs_quantized_static(target_planes, depths_m, wavelength, dx, n_iters, bits, scales,
                                    device="cuda", seed=0, pad_factor=2.0):
    shape = target_planes[0].shape
    targets_amp = _to_tensor_targets(target_planes, device)

    torch.manual_seed(seed)
    slm_phase = (torch.rand(shape, device=device, dtype=torch.float32) * 2 * math.pi - math.pi)
    slm_field = quantize_static(torch.exp(1j * slm_phase), bits, scales["slm_field"])

    history = []
    per_plane_history = [[] for _ in target_planes]

    for it in range(1, n_iters + 1):
        correction = torch.zeros(shape, dtype=torch.complex64, device=device)
        recon_planes = []
        for target_amp, z in zip(targets_amp, depths_m):
            field_at_plane = angular_spectrum_propagate(slm_field, wavelength, dx, z, pad_factor=pad_factor)
            field_at_plane = quantize_static(field_at_plane, bits, scales["field_at_plane"])
            recon_planes.append(safe_abs(field_at_plane).detach())

            phase_at_plane = torch.angle(field_at_plane)
            constrained = target_amp.to(torch.complex64) * torch.exp(1j * phase_at_plane.to(torch.float32))
            constrained = quantize_static(constrained, bits, scales["constrained"])

            back = angular_spectrum_propagate(constrained, wavelength, dx, -z, pad_factor=pad_factor)
            back = quantize_static(back, bits, scales["back"])
            correction += back

        slm_field = correction / len(depths_m)
        slm_phase = torch.angle(slm_field)
        slm_field = quantize_static(torch.exp(1j * slm_phase), bits, scales["slm_field"])

        psnrs = [_psnr_torch(r, t) for r, t in zip(recon_planes, targets_amp)]
        history.append(sum(psnrs) / len(psnrs))
        for i, v in enumerate(psnrs):
            per_plane_history[i].append(v)

    return history, per_plane_history


def find_plateau(history, tol=PLATEAU_TOLERANCE_DB):
    final = history[-1]
    plateau_iter = next(i + 1 for i, v in enumerate(history) if v >= final - tol)
    still_rising = len(history) >= 10 and (final - history[-10]) > tol
    return plateau_iter, final, still_rising


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

    print(f"Calibrating static scales ({CALIBRATION_ITERS} iterations, full resolution)...")
    scales = calibrate_scales(targets, DEPTHS_M, WAVELENGTH, DX, CALIBRATION_ITERS,
                               device=DEVICE, pad_factor=PAD_FACTOR)
    for slot, val in scales.items():
        print(f"  {slot}: max magnitude = {val:.6g}")
    print()

    results = {}
    for bits in BITS_SWEEP:
        label = "float32 (baseline)" if bits is None else f"{bits}-bit"
        print(f"Running {label} (static scale, up to {N_ITERS_MAX} iterations, plateau auto-detected)...")
        t0 = time.time()
        history, per_plane = multiplane_gs_quantized_static(
            targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS_MAX, bits=bits, scales=scales,
            device=DEVICE, pad_factor=PAD_FACTOR
        )
        elapsed = time.time() - t0
        plateau_iter, final_psnr, still_rising = find_plateau(history)
        results[label] = {"history": history, "per_plane": per_plane, "elapsed": elapsed,
                           "plateau_iter": plateau_iter, "still_rising": still_rising}
        final_per_plane = [ph[-1] for ph in per_plane]
        rising_flag = " STILL RISING" if still_rising else ""
        print(f"   plateau@{plateau_iter}/{N_ITERS_MAX}{rising_flag}, final avg: {final_psnr:.2f} dB | "
              f"near/mid/far: {final_per_plane[0]:.2f}/{final_per_plane[1]:.2f}/{final_per_plane[2]:.2f} dB | "
              f"{elapsed:.1f}s")

    baseline_label = "float32 (baseline)"
    baseline_final = results[baseline_label]["history"][-1]
    baseline_per_plane = [ph[-1] for ph in results[baseline_label]["per_plane"]]

    print("\n" + "=" * 100)
    print(f"{'Bits':<20}{'Avg dB':<12}{'Delta vs fp32':<16}{'Near D':<10}{'Mid D':<10}{'Far D':<10}{'Plateau iter':<14}")
    int8_delta = None
    any_rising = False
    for bits in BITS_SWEEP:
        label = "float32 (baseline)" if bits is None else f"{bits}-bit"
        r = results[label]
        final = r["history"][-1]
        final_pp = [ph[-1] for ph in r["per_plane"]]
        delta = final - baseline_final
        deltas_pp = [final_pp[i] - baseline_per_plane[i] for i in range(3)]
        any_rising = any_rising or r["still_rising"]
        print(f"{label:<20}{final:<12.2f}{delta:<+16.2f}{deltas_pp[0]:<+10.2f}{deltas_pp[1]:<+10.2f}"
              f"{deltas_pp[2]:<+10.2f}{r['plateau_iter']:<14}")
        if bits == 8:
            int8_delta = delta
    print("=" * 100)

    if any_rising:
        print("\nWARNING: at least one condition was still rising at N_ITERS_MAX -- increase it before")
        print("fully trusting the numbers above.")

    if int8_delta is not None:
        print(f"\n8-bit (INT8-equivalent), STATIC scale, FULL RESOLUTION, TRUE CONVERGENCE, "
              f"delta vs float32: {int8_delta:+.2f} dB")
        if int8_delta > -0.5:
            print("Survives the jump to full resolution and true convergence -- unlike 10.3's sequential-")
            print("rendering finding, this is NOT a small-scale artifact. Real evidence Section 4.5's")
            print("INT8-corrupts-the-computation assumption doesn't hold for this algorithm/content.")
        elif int8_delta > -2.0:
            print("Moderate degradation appears at full resolution/true convergence that wasn't visible")
            print("at 512x512/16 iterations -- matches the general pattern of small-scale findings")
            print("shrinking or changing at scale. Worth characterizing further before trusting either way.")
        else:
            print("Severe degradation at full resolution/true convergence -- the small-scale result was")
            print("misleading. Section 4.5's original INT8 assumption is supported after all at this scale.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(8, 5))
    bit_labels = [b if b is not None else 20 for b in BITS_SWEEP]
    avg_vals = [results["float32 (baseline)" if b is None else f"{b}-bit"]["history"][-1] for b in BITS_SWEEP]
    near_vals = [results["float32 (baseline)" if b is None else f"{b}-bit"]["per_plane"][0][-1] for b in BITS_SWEEP]
    mid_vals = [results["float32 (baseline)" if b is None else f"{b}-bit"]["per_plane"][1][-1] for b in BITS_SWEEP]
    far_vals = [results["float32 (baseline)" if b is None else f"{b}-bit"]["per_plane"][2][-1] for b in BITS_SWEEP]

    ax.plot(bit_labels, avg_vals, label="Average (all planes)", marker="o", linewidth=2)
    ax.plot(bit_labels, near_vals, label="Near (sparse)", marker="^", linestyle="--", alpha=0.7)
    ax.plot(bit_labels, mid_vals, label="Mid (sparse)", marker="s", linestyle="--", alpha=0.7)
    ax.plot(bit_labels, far_vals, label="Far (dense)", marker="D", linestyle="--", alpha=0.7)
    ax.axvline(8, color="gray", linestyle=":", label="INT8 (8 bits)")
    ax.set_xlabel("Bit depth (rightmost point = float32 baseline)")
    ax.set_ylabel("PSNR (dB) at true convergence")
    ax.set_title(f"Quantization emulation (static scale, full-res, true convergence)\n"
                 f"{SHAPE[1]}x{SHAPE[0]}, real content")
    ax.legend()
    plt.tight_layout()
    plt.savefig("int8_precision_fullres_converged.png", dpi=130)
    print("Saved plot: int8_precision_fullres_converged.png")


if __name__ == "__main__":
    main()
