"""
10.8 item 1: confirm the color finding (10.7) at full resolution and
across all three depth planes, and test whether jointly optimizing one
phase pattern across all colors AND depths at once narrows the 5-6dB
gap found at small scale, before assuming three fully independent
per-color solves are required.

Three conditions, same realistic content (near/mid/far) used in 10.6:

  A) Native: solve each color independently (green, red, blue), each
     one a normal 3-plane simultaneous GS solve. This is the upper
     bound -- best achievable quality if you're willing to pay for
     three fully separate solves.

  B) Naive reuse: take the green-native phase from (A), propagate it
     unmodified at red and blue wavelengths, no re-solving. This is the
     lower bound / worst case -- what happens if you do nothing extra
     for color.

  C) Joint: ONE phase pattern, optimized against all 9 (depth, color)
     constraints simultaneously via multiconstraint_gs_torch. Tests
     whether joint optimization buys back some of the (A)-(B) gap at no
     extra compute cost (same total FFT count either way, same
     principle as the depth-plane grouping test in 10.3), or whether it
     hits its own ceiling the way simultaneous multi-plane depth did.

Run on your 3060:
    python3 experiment_color_full_resolution.py
"""
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch, multiconstraint_gs_torch
from propagation_torch import angular_spectrum_propagate, safe_abs
from metrics import psnr as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 16

WAVELENGTHS = {"green": 520e-9, "red": 638e-9, "blue": 450e-9}
PLANE_NAMES = ["near", "mid", "far"]


def run_and_time(fn, *args, **kwargs):
    counter.reset()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    result = fn(*args, **kwargs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return result, elapsed, counter.count


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    targets = make_realistic_multiplane_target(SHAPE)

    # warm-up
    print("Warming up GPU...")
    _ = run_and_time(multiplane_gs_torch, targets, DEPTHS_M, WAVELENGTHS["green"], DX, 1,
                      device=DEVICE, pad_factor=PAD_FACTOR)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- A: native per-color solves ----
    print("\nA) Native per-color solves (3 runs, one per color)...")
    native_quality = {}   # native_quality[color][plane_idx]
    native_phase = {}
    for color, wl in WAVELENGTHS.items():
        (phase, hist, cps), t, ffts = run_and_time(
            multiplane_gs_torch, targets, DEPTHS_M, wl, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR
        )
        final = max(cps.keys())
        q = [psnr_np(cps[final][i], targets[i]) for i in range(3)]
        native_quality[color] = q
        native_phase[color] = phase
        print(f"   {color}: {[round(x,1) for x in q]} dB  ({t:.2f}s, {ffts} FFTs)")

    # ---- B: naive cross-wavelength reuse of the green phase ----
    print("\nB) Naive reuse: green-native phase, propagated at red/blue (no re-solving)...")
    reuse_quality = {"green": native_quality["green"]}
    for color in ["red", "blue"]:
        wl = WAVELENGTHS[color]
        q = []
        for i, z in enumerate(DEPTHS_M):
            recon = angular_spectrum_propagate(torch.exp(1j * native_phase["green"]), wl, DX, z,
                                                pad_factor=PAD_FACTOR)
            q.append(psnr_np(safe_abs(recon).cpu().numpy(), targets[i]))
        reuse_quality[color] = q
        print(f"   {color}: {[round(x,1) for x in q]} dB")

    # ---- C: joint multi-wavelength, multi-depth solve ----
    print("\nC) Joint solve: one phase pattern, all 9 (depth, color) constraints at once...")
    constraints = []
    for i, z in enumerate(DEPTHS_M):
        for color, wl in WAVELENGTHS.items():
            constraints.append({"target": targets[i], "z": z, "wavelength": wl})
    (joint_phase, joint_hist, joint_cps), t_joint, ffts_joint = run_and_time(
        multiconstraint_gs_torch, constraints, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR
    )
    final_joint = max(joint_cps.keys())
    joint_quality = {"green": [], "red": [], "blue": []}
    idx = 0
    for i in range(3):
        for color in WAVELENGTHS:
            q = psnr_np(joint_cps[final_joint][idx], targets[i])
            joint_quality[color].append(q)
            idx += 1
    print(f"   ({t_joint:.2f}s, {ffts_joint} FFTs)")
    for color in WAVELENGTHS:
        print(f"   {color}: {[round(x,1) for x in joint_quality[color]]} dB")

    # ---- summary ----
    print("\n" + "=" * 78)
    print(f"{'plane':<8}{'color':<8}{'A: native':<12}{'B: naive reuse':<16}{'C: joint':<10}")
    for i, pname in enumerate(PLANE_NAMES):
        for color in WAVELENGTHS:
            a = native_quality[color][i]
            b = reuse_quality[color][i]
            c = joint_quality[color][i]
            print(f"{pname:<8}{color:<8}{a:<12.1f}{b:<16.1f}{c:<10.1f}")
    print("=" * 78)
    avg_a = np.mean([native_quality[c][i] for c in WAVELENGTHS for i in range(3)])
    avg_b_nongreen = np.mean([reuse_quality[c][i] for c in ["red", "blue"] for i in range(3)])
    avg_c_nongreen = np.mean([joint_quality[c][i] for c in ["red", "blue"] for i in range(3)])
    print(f"Average native (A): {avg_a:.1f} dB")
    print(f"Average naive reuse, red+blue only (B): {avg_b_nongreen:.1f} dB")
    print(f"Average joint solve, red+blue only (C): {avg_c_nongreen:.1f} dB")
    if avg_c_nongreen > avg_b_nongreen + 1.0:
        print("Joint optimization meaningfully narrows the gap vs. naive reuse -- worth pursuing")
        print("over three fully independent per-color solves.")
    elif avg_c_nongreen < avg_a - 3.0:
        print("Joint solve is well below native per-color quality -- similar pattern to the")
        print("simultaneous multi-plane depth finding: packing more constraints into one phase")
        print("pattern costs quality. Three independent per-color solves may be the only path")
        print("to native-level color quality.")
    else:
        print("Joint solve lands close to native -- promising, worth further testing.")
    if DEVICE == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, pname in enumerate(PLANE_NAMES):
        axes[0, i].imshow(targets[i], cmap="gray")
        axes[0, i].set_title(f"Target: {pname}")
        axes[0, i].axis("off")

    # bottom two rows: naive reuse vs joint, for the RED channel (most illustrative)
    for i, pname in enumerate(PLANE_NAMES):
        wl = WAVELENGTHS["red"]
        z = DEPTHS_M[i]
        recon_reuse = angular_spectrum_propagate(torch.exp(1j * native_phase["green"]), wl, DX, z,
                                                   pad_factor=PAD_FACTOR)
        axes[1, i].imshow(safe_abs(recon_reuse).cpu().numpy(), cmap="gray")
        axes[1, i].set_title(f"B: naive reuse (red), {pname} ({reuse_quality['red'][i]:.1f} dB)")
        axes[1, i].axis("off")
    idx = 0
    for i in range(3):
        for color in WAVELENGTHS:
            if color == "red":
                axes[2, i].imshow(joint_cps[final_joint][idx], cmap="gray")
                axes[2, i].set_title(f"C: joint (red), {PLANE_NAMES[i]} ({joint_quality['red'][i]:.1f} dB)")
                axes[2, i].axis("off")
            idx += 1
    plt.tight_layout()
    plt.savefig("color_full_resolution_results.png", dpi=130)
    print("Saved plot: color_full_resolution_results.png")


if __name__ == "__main__":
    main()
