"""
Audit follow-up (10.9): the ORIGINAL two-plane dropoff test (this
project's standard near/mid/far content -- icon/text/photo-scene, not
the CLOSEUP/LAYERED profiles built later for the content-dependence
investigation) was only ever run at 512x512. The CLOSEUP/LAYERED variants
did get a full-resolution retest (and that retest reversed the
content-adaptivity claim built on top of them), but the base scenario
itself -- does dropping mid lose real content, and does it save real
compute, for this project's actual standard content -- was never itself
confirmed at the document's real target resolution.

Same three conditions as the original test, same masked-PSNR correction
already applied to it (mid is sparse text content), just at full
resolution instead of 512x512.

Run on your 3060 (expect several minutes):
    python3 experiment_two_plane_dropoff_fullres.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, sparse_content_bounds
from propagation_torch import angular_spectrum_propagate, safe_abs
from retrieval_torch import multiplane_gs_torch, _to_tensor_targets, _psnr_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)             # full target resolution, up from 512x512
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
MID_INDEX = 1
N_ITERS_BUDGET = 250
PLATEAU_EPS_DB = 0.1


def find_plateau_robust(history, eps=PLATEAU_EPS_DB):
    final = history[-1]
    for i in range(len(history)):
        if all(abs(v - final) < eps for v in history[i:]):
            return i + 1
    return len(history)


def solve_multiplane(targets_subset, depths_subset, seed=0):
    t0 = time.time()
    phase, history, _ = multiplane_gs_torch(
        targets_subset, depths_subset, WAVELENGTH, DX, N_ITERS_BUDGET,
        device=DEVICE, seed=seed, pad_factor=PAD_FACTOR, smooth_cutoff=True
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    plateau = find_plateau_robust(history)
    still_rising = len(history) >= 20 and (history[-1] - sum(history[-20:-10]) / 10) > PLATEAU_EPS_DB
    ffts_per_iter = 4 * len(depths_subset)
    return {"phase": phase, "history": history, "plateau": plateau,
            "final": history[-1], "ffts_at_plateau": plateau * ffts_per_iter,
            "elapsed": elapsed, "still_rising": still_rising}


def per_plane_quality(phase, targets_amp, depths):
    results = []
    for target_amp, z in zip(targets_amp, depths):
        field = torch.exp(1j * phase.to(torch.float32))
        recon = safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR))
        results.append(_psnr_torch(recon, target_amp))
    return results


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)
    targets_amp = _to_tensor_targets(targets, DEVICE)
    near, mid, far = targets
    rows, cols = sparse_content_bounds(mid)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print("\nA) 3-plane baseline (near, mid, far)...")
    baseline = solve_multiplane(targets, DEPTHS_M, seed=0)
    baseline_per_plane_whole = per_plane_quality(baseline["phase"], targets_amp, DEPTHS_M)
    baseline_mid_recon = safe_abs(angular_spectrum_propagate(
        torch.exp(1j * baseline["phase"].to(torch.float32)), WAVELENGTH, DX, DEPTHS_M[MID_INDEX], pad_factor=PAD_FACTOR
    ))
    baseline_mid_masked = _psnr_torch(baseline_mid_recon[rows, cols], targets_amp[MID_INDEX][rows, cols])
    print(f"   plateau={baseline['plateau']}, FFTs/frame={baseline['ffts_at_plateau']}, "
          f"{baseline['elapsed']:.1f}s{' STILL RISING' if baseline['still_rising'] else ''}")
    print(f"   per-plane (whole-frame): near={baseline_per_plane_whole[0]:.2f} "
          f"mid={baseline_per_plane_whole[1]:.2f} far={baseline_per_plane_whole[2]:.2f} dB")
    print(f"   mid (masked, correct for sparse text): {baseline_mid_masked:.2f} dB")

    print("\nB) 2-plane (near, far only -- mid dropped)...")
    two_plane_targets = [near, far]
    two_plane_depths = [DEPTHS_M[0], DEPTHS_M[2]]
    two_plane = solve_multiplane(two_plane_targets, two_plane_depths, seed=0)
    print(f"   plateau={two_plane['plateau']}, FFTs/frame={two_plane['ffts_at_plateau']}, "
          f"{two_plane['elapsed']:.1f}s{' STILL RISING' if two_plane['still_rising'] else ''}")

    print("\nC) Proxy: propagate B's phase to the dropped mid depth...")
    mid_recon = safe_abs(angular_spectrum_propagate(
        torch.exp(1j * two_plane["phase"].to(torch.float32)), WAVELENGTH, DX, DEPTHS_M[MID_INDEX], pad_factor=PAD_FACTOR
    ))
    dropped_mid_quality = _psnr_torch(mid_recon[rows, cols], targets_amp[MID_INDEX][rows, cols])

    blank_field = torch.zeros_like(targets_amp[MID_INDEX])
    blank_floor = _psnr_torch(blank_field[rows, cols], targets_amp[MID_INDEX][rows, cols])

    ffts_saved_pct = 100 * (1 - two_plane["ffts_at_plateau"] / baseline["ffts_at_plateau"])
    gap_to_ceiling = baseline_mid_masked - dropped_mid_quality
    gap_to_floor = dropped_mid_quality - blank_floor

    print(f"\n   dropped-mid proxy quality (masked): {dropped_mid_quality:.2f} dB")
    print(f"   blank floor (masked): {blank_floor:.2f} dB")
    print(f"   FFTs saved by dropping mid: {ffts_saved_pct:+.1f}%")
    print(f"   gap to ceiling: {gap_to_ceiling:.2f} dB, gap to floor: {gap_to_floor:.2f} dB")

    print("\n" + "=" * 78)
    print(f"Compare against the original 512x512 result:")
    print(f"  512x512:      ceiling=9.76dB, proxy=6.29dB, floor=4.86dB, FFTs saved=+22.7%")
    print(f"  Full res:     ceiling={baseline_mid_masked:.2f}dB, proxy={dropped_mid_quality:.2f}dB, "
          f"floor={blank_floor:.2f}dB, FFTs saved={ffts_saved_pct:+.1f}%")
    print("=" * 78)

    if gap_to_floor > 0.5 and ffts_saved_pct > 5:
        print("\nBase finding holds at full resolution: dropping mid still saves real compute and the")
        print("leaked signal at the dropped depth is still meaningfully above the blank floor -- the core")
        print("10.8 claim (not the retracted content-adaptivity extension) is confirmed at target resolution.")
    else:
        print("\nBase finding does NOT clearly hold at full resolution -- report this divergence directly.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mid, cmap="gray"); axes[0].set_title("Real mid content"); axes[0].axis("off")
    axes[1].imshow(mid_recon.detach().cpu().numpy(), cmap="gray")
    axes[1].set_title(f"2-plane proxy ({dropped_mid_quality:.1f} dB)"); axes[1].axis("off")
    axes[2].imshow(baseline_mid_recon.detach().cpu().numpy(), cmap="gray")
    axes[2].set_title(f"3-plane actual ({baseline_mid_masked:.1f} dB)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("two_plane_dropoff_fullres_base.png", dpi=130)
    print("\nSaved plot: two_plane_dropoff_fullres_base.png")


if __name__ == "__main__":
    main()
