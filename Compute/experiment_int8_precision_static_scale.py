"""
Follow-up to experiment_int8_precision.py.

That script's quantization emulation rescales to each tensor's own max
magnitude at EVERY quantize call -- an oracle-perfect adaptive scale a
real ASIC datapath doesn't get. It found essentially zero degradation at
8 bits, which is suspiciously clean for a claim as consequential as
overturning Section 4.5's "INT8 corrupts the computation" assumption.
Before trusting that, this reruns the same sweep with a STATIC scale:
calibrated once (from the unquantized float32 baseline's observed
worst-case dynamic range at each of the four quantization points, across
all 16 iterations), then reused unchanged for every iteration and every
bit-depth run. This is much closer to how a real fixed-point datapath
actually works -- a calibrated format chosen once, not re-normalized
every operation -- and will surface any degradation the adaptive-scale
version's oracle rescaling was hiding.

Same GS structure, same content, same 16-iteration fixed budget, same
bit-depth sweep as the adaptive-scale version, so the two are directly
comparable side by side.

Run on your 3060:
    python3 experiment_int8_precision_static_scale.py
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
SHAPE = (512, 512)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 16
BITS_SWEEP = [4, 6, 8, 10, 12, 16, None]
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
    """
    Run the unquantized (float32) GS loop once, recording the worst-case
    (max across all iterations and all planes) magnitude observed at each
    of the four quantization insertion points. This is the calibration a
    real ASIC's fixed-point format would need to be designed around --
    covering the observed dynamic range without clipping, chosen once.
    """
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


def main():
    print(f"Device: {DEVICE}")
    targets = make_realistic_multiplane_target(SHAPE)

    print("Calibrating static scales from the unquantized (float32) baseline run...")
    scales = calibrate_scales(targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR)
    for slot, val in scales.items():
        print(f"  {slot}: max magnitude = {val:.6g}")
    print()

    results = {}
    for bits in BITS_SWEEP:
        label = "float32 (baseline)" if bits is None else f"{bits}-bit"
        print(f"Running {label} (static scale)...")
        t0 = time.time()
        history, per_plane = multiplane_gs_quantized_static(
            targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, bits=bits, scales=scales,
            device=DEVICE, pad_factor=PAD_FACTOR
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
    print(f"{'Bits':<20}{'Avg dB':<12}{'Delta vs fp32':<16}{'Near D':<10}{'Mid D':<10}{'Far D':<10}")
    int8_delta = None
    for bits in BITS_SWEEP:
        label = "float32 (baseline)" if bits is None else f"{bits}-bit"
        final = results[label]["history"][-1]
        final_pp = [ph[-1] for ph in results[label]["per_plane"]]
        delta = final - baseline_final
        deltas_pp = [final_pp[i] - baseline_per_plane[i] for i in range(3)]
        print(f"{label:<20}{final:<12.2f}{delta:<+16.2f}{deltas_pp[0]:<+10.2f}{deltas_pp[1]:<+10.2f}{deltas_pp[2]:<+10.2f}")
        if bits == 8:
            int8_delta = delta
    print("=" * 90)

    if int8_delta is not None:
        print(f"\n8-bit (INT8-equivalent), STATIC scale, delta vs float32 baseline: {int8_delta:+.2f} dB")
        if int8_delta > -0.5:
            print("Still negligible degradation even with a realistic static (not per-call adaptive) scale --")
            print("the adaptive-scale version's clean result was NOT just an artifact of oracle rescaling.")
        elif int8_delta > -2.0:
            print("Moderate degradation appears once the scale is fixed rather than adaptive -- the adaptive")
            print("version's clean result WAS partly an artifact of oracle-perfect rescaling per call.")
        else:
            print("Severe degradation under a static scale -- confirms the adaptive-scale version's clean")
            print("result was misleading: a real fixed-point datapath would see real corruption at 8 bits.")

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
    ax.set_ylabel("PSNR (dB), 16-iteration fixed budget")
    ax.set_title(f"Quantization emulation (STATIC scale): PSNR vs. bit depth, {SHAPE[0]}x{SHAPE[1]}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("int8_precision_sweep_static_scale.png", dpi=130)
    print("\nSaved plot: int8_precision_sweep_static_scale.png")


if __name__ == "__main__":
    main()
