"""
Section 4.5 states that INT8 (20 TOPS/W) "corrupts the wave computation"
and requires BFP16 (6.5 TOPS/W) instead -- but this has never actually
been tested against real content. It's an assumption, not a measurement.
If it holds even partially, it's a direct ~3x lever on Gap 2's power
figure (20 vs 6.5 TOPS/W).

METHODOLOGY CAVEAT -- read before trusting the results:
This GPU has no native fixed-point INT8 FFT path, so true hardware
behavior can't be measured directly. What this script does instead is
quantization EMULATION: round the field to N-bit fixed-point precision
at each point in the GS loop where a real ASIC's datapath would actually
be bit-limited (after each forward propagate, after the amplitude
constraint, after each backward propagate, and on the accumulated SLM
field), then continue the surrounding arithmetic in float32. This
approximates the CUMULATIVE effect of running at N bits, but does NOT
model how rounding error compounds INSIDE a single FFT's internal
butterfly stages -- that would require a from-scratch fixed-point FFT
implementation, a much bigger undertaking. Treat results here as a
reasonable first-order signal, not a hardware-validated verdict.

Uses a fixed 16-iteration budget for fast relative comparison across bit
depths -- NOT full convergence (10.1c/10.1 found true convergence needs
far more). Per this document's own established practice (10.1d), fixed-
budget comparisons are still valid for RELATIVE findings (does quality
drop between bit depths?) even though they understate absolute quality.
If a real cliff is found, it's worth rerunning at true convergence next.

Sweeps bit depth from 4 to 16 (float32, effectively unquantized, as the
reference baseline) and reports PSNR per plane -- near/mid are closer to
the sparse UI content in 10.6, far is the dense photo-like scene -- to
check whether the corruption is uniform or content-dependent, as
speculated when this test was proposed.

Run on your 3060:
    python3 experiment_int8_precision.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from propagation_torch import angular_spectrum_propagate, safe_abs
from retrieval_torch import _to_tensor_targets, _psnr_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)          # small scale first, matches this project's usual pattern
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 16                 # fixed budget for fast relative comparison -- see docstring
BITS_SWEEP = [4, 6, 8, 10, 12, 16, None]   # None = float32 baseline, no quantization
PLANE_NAMES = ["near", "mid", "far"]       # near/mid ~ sparse UI content, far ~ dense photo-like scene


def quantize_dequantize(field, bits):
    """
    Simulated symmetric per-tensor fixed-point quantization of a complex
    field to `bits`-bit precision, then immediately converted back to
    float for continued float-domain computation (standard "fake
    quantization" emulation). bits=None is a no-op (float32 passthrough).
    """
    if bits is None:
        return field
    real, imag = field.real, field.imag
    max_val = torch.maximum(real.abs().max(), imag.abs().max()).clamp(min=1e-12)
    levels = 2 ** (bits - 1) - 1
    scale = max_val / levels
    real_q = torch.round(real / scale) * scale
    imag_q = torch.round(imag / scale) * scale
    return torch.complex(real_q, imag_q)


def multiplane_gs_quantized(target_planes, depths_m, wavelength, dx, n_iters, bits, device="cuda", seed=0,
                             pad_factor=2.0):
    """
    Same GS structure as multiplane_gs_torch, with quantize_dequantize
    inserted at every point a real fixed-point ASIC datapath would
    actually round: after each forward propagate, after the amplitude
    constraint, after each backward propagate, and on the accumulated
    SLM field between iterations.
    """
    import math
    shape = target_planes[0].shape
    targets_amp = _to_tensor_targets(target_planes, device)

    torch.manual_seed(seed)
    slm_phase = (torch.rand(shape, device=device, dtype=torch.float32) * 2 * math.pi - math.pi)
    slm_field = quantize_dequantize(torch.exp(1j * slm_phase), bits)

    history = []
    per_plane_history = [[] for _ in target_planes]

    for it in range(1, n_iters + 1):
        correction = torch.zeros(shape, dtype=torch.complex64, device=device)
        recon_planes = []
        for target_amp, z in zip(targets_amp, depths_m):
            field_at_plane = angular_spectrum_propagate(slm_field, wavelength, dx, z, pad_factor=pad_factor)
            field_at_plane = quantize_dequantize(field_at_plane, bits)
            recon_planes.append(safe_abs(field_at_plane).detach())

            phase_at_plane = torch.angle(field_at_plane)
            constrained = target_amp.to(torch.complex64) * torch.exp(1j * phase_at_plane.to(torch.float32))
            constrained = quantize_dequantize(constrained, bits)

            back = angular_spectrum_propagate(constrained, wavelength, dx, -z, pad_factor=pad_factor)
            back = quantize_dequantize(back, bits)
            correction += back

        slm_field = correction / len(depths_m)
        slm_phase = torch.angle(slm_field)
        slm_field = quantize_dequantize(torch.exp(1j * slm_phase), bits)

        psnrs = [_psnr_torch(r, t) for r, t in zip(recon_planes, targets_amp)]
        history.append(sum(psnrs) / len(psnrs))
        for i, v in enumerate(psnrs):
            per_plane_history[i].append(v)

    return history, per_plane_history


def main():
    print(f"Device: {DEVICE}")
    print("NOTE: this is a quantization EMULATION, not native fixed-point hardware -- see module docstring.\n")

    targets = make_realistic_multiplane_target(SHAPE)

    results = {}
    for bits in BITS_SWEEP:
        label = "float32 (baseline)" if bits is None else f"{bits}-bit"
        print(f"Running {label}...")
        t0 = time.time()
        history, per_plane = multiplane_gs_quantized(
            targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, bits=bits, device=DEVICE, pad_factor=PAD_FACTOR
        )
        elapsed = time.time() - t0
        results[label] = {"history": history, "per_plane": per_plane, "elapsed": elapsed}
        final_per_plane = [ph[-1] for ph in per_plane]
        print(f"   final avg: {history[-1]:.2f} dB | near/mid/far: "
              f"{final_per_plane[0]:.2f}/{final_per_plane[1]:.2f}/{final_per_plane[2]:.2f} dB | {elapsed:.2f}s")

    baseline_label = "float32 (baseline)"
    baseline_final = results[baseline_label]["history"][-1]
    baseline_per_plane = [ph[-1] for ph in results[baseline_label]["per_plane"]]

    print("\n" + "=" * 90)
    print(f"{'Bits':<20}{'Avg dB':<12}{'Δ vs fp32':<14}{'Near Δ':<10}{'Mid Δ':<10}{'Far Δ':<10}")
    int8_delta = None
    for bits in BITS_SWEEP:
        label = "float32 (baseline)" if bits is None else f"{bits}-bit"
        final = results[label]["history"][-1]
        final_pp = [ph[-1] for ph in results[label]["per_plane"]]
        delta = final - baseline_final
        deltas_pp = [final_pp[i] - baseline_per_plane[i] for i in range(3)]
        print(f"{label:<20}{final:<12.2f}{delta:<+14.2f}{deltas_pp[0]:<+10.2f}{deltas_pp[1]:<+10.2f}{deltas_pp[2]:<+10.2f}")
        if bits == 8:
            int8_delta = delta
    print("=" * 90)

    if int8_delta is not None:
        print(f"\n8-bit (INT8-equivalent) delta vs float32 baseline: {int8_delta:+.2f} dB")
        if int8_delta > -0.5:
            print("Negligible degradation at 8 bits -- Section 4.5's 'INT8 corrupts the computation' assumption")
            print("does NOT hold up at this precision, at least for this content and this quantization model.")
            print("Worth escalating: rerun at true convergence and at full resolution before revising Gap 2's")
            print("power figure (20 TOPS/W instead of 6.5 would cut the power number roughly 3x).")
        elif int8_delta > -2.0:
            print("Moderate degradation at 8 bits -- not clean, not catastrophic. Worth checking where exactly")
            print("the curve bends (see plot) to see if a slightly higher bit depth (10-12) recovers most of the")
            print("loss cheaply, which would still beat BFP16's 6.5 TOPS/W.")
        else:
            print("Severe degradation at 8 bits -- supports Section 4.5's original assumption. The bend point in")
            print("the plot below is the more useful number going forward than a binary INT8 verdict.")

    fig, ax = plt.subplots(figsize=(8, 5))
    bit_labels = [b if b is not None else 20 for b in BITS_SWEEP]  # plot float32 baseline off to the right
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
    ax.set_ylabel("PSNR (dB), 16-iteration fixed budget")
    ax.set_title(f"Quantization emulation: PSNR vs. bit depth, {SHAPE[0]}x{SHAPE[1]}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("int8_precision_sweep.png", dpi=130)
    print("\nSaved plot: int8_precision_sweep.png")


if __name__ == "__main__":
    main()
