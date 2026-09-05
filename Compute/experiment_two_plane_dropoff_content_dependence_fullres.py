"""
Full-resolution retest of experiment_two_plane_dropoff_content_dependence.py.

That 512x512 run found (after correcting a measurement confound with a
normalized fraction-of-available-range metric) that a CLOSEUP scene
recovers more of its dropped mid-plane content for free (38.7%) than a
LAYERED, street-scene-style scene does (25.8%) -- supporting
content-adaptive plane count. This project's whole session has shown
small-scale findings repeatedly need a full-resolution check before
they're trustworthy (10.3's sequential-rendering advantage vanished at
scale; donor-seeding reversed; scene-cut negative transfer didn't
replicate) -- this closes that gap for the plane-dropoff/content-
dependence finding specifically.

Same two content profiles (CLOSEUP: dominant near, sparse mid, flat far;
LAYERED: substantial distinct content at all three depths), same
final-value-anchored plateau detector, same normalized
fraction-of-available-range metric -- only the resolution changes, from
512x512 to the document's actual 2,700x4,700 target.

N_ITERS_BUDGET raised to 250 (from 150 at small scale) -- full-resolution
GS has needed up to ~200 iterations in worst cases seen elsewhere this
session, so 150 would risk under-counting convergence here.

Run on your 3060 (expect several minutes):
    python3 experiment_two_plane_dropoff_content_dependence_fullres.py
"""
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, make_multiplane_target, sparse_content_bounds
from propagation_torch import angular_spectrum_propagate, safe_abs
from retrieval_torch import multiplane_gs_torch, _to_tensor_targets, _psnr_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)            # full target resolution, up from 512x512
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
MID_INDEX = 1
N_ITERS_BUDGET = 250              # up from 150 -- see docstring
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
    ffts_per_iter = 4 * len(depths_subset)
    still_rising = len(history) >= 20 and (history[-1] - sum(history[-20:-10]) / 10) > PLATEAU_EPS_DB
    return {"phase": phase, "history": history, "plateau": plateau,
            "final": history[-1], "ffts_at_plateau": plateau * ffts_per_iter, "elapsed": elapsed,
            "still_rising": still_rising}


def make_closeup_scene(shape):
    base = make_realistic_multiplane_target(shape)
    near = base[0]

    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    cy, cx = ny * 0.7, nx * 0.7
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mid = (r < min(ny, nx) * 0.03).astype(float)

    far = np.full(shape, 0.15) + 0.03 * np.random.default_rng(1).standard_normal(shape)
    far = np.clip(far, 0, 1)

    return [near, mid, far]


def make_layered_scene(shape):
    base = make_realistic_multiplane_target(shape)
    near = base[0]
    mid = base[2]

    far_planes = make_multiplane_target(shape, n_planes=3, soft=True, sigma=8.0)
    far = far_planes[1]

    return [near, mid, far]


def evaluate_profile(name, scene):
    print("\n" + "=" * 90)
    print(f"PROFILE: {name}")
    print("=" * 90)
    near, mid, far = scene
    targets_amp = _to_tensor_targets(scene, DEVICE)
    rows, cols = sparse_content_bounds(mid)
    print(f"  mid content bounding box: rows={rows}, cols={cols}")

    baseline = solve_multiplane(scene, DEPTHS_M, seed=0)
    baseline_mid_recon = safe_abs(angular_spectrum_propagate(
        torch.exp(1j * baseline["phase"].to(torch.float32)), WAVELENGTH, DX, DEPTHS_M[MID_INDEX], pad_factor=PAD_FACTOR
    ))
    baseline_mid_masked = _psnr_torch(baseline_mid_recon[rows, cols], targets_amp[MID_INDEX][rows, cols])
    print(f"  3-plane baseline: plateau={baseline['plateau']}, FFTs/frame={baseline['ffts_at_plateau']}, "
          f"mid quality (masked)={baseline_mid_masked:.2f} dB{' STILL RISING' if baseline['still_rising'] else ''}, "
          f"{baseline['elapsed']:.1f}s")

    two_plane_targets = [near, far]
    two_plane_depths = [DEPTHS_M[0], DEPTHS_M[2]]
    two_plane = solve_multiplane(two_plane_targets, two_plane_depths, seed=0)
    print(f"  2-plane (mid dropped): plateau={two_plane['plateau']}, FFTs/frame={two_plane['ffts_at_plateau']}"
          f"{' STILL RISING' if two_plane['still_rising'] else ''}, {two_plane['elapsed']:.1f}s")

    mid_recon = safe_abs(angular_spectrum_propagate(
        torch.exp(1j * two_plane["phase"].to(torch.float32)), WAVELENGTH, DX, DEPTHS_M[MID_INDEX], pad_factor=PAD_FACTOR
    ))
    dropped_mid_quality = _psnr_torch(mid_recon[rows, cols], targets_amp[MID_INDEX][rows, cols])

    blank_field = torch.zeros_like(targets_amp[MID_INDEX])
    blank_floor = _psnr_torch(blank_field[rows, cols], targets_amp[MID_INDEX][rows, cols])

    ffts_saved_pct = 100 * (1 - two_plane["ffts_at_plateau"] / baseline["ffts_at_plateau"])
    gap_to_ceiling = baseline_mid_masked - dropped_mid_quality
    gap_to_floor = dropped_mid_quality - blank_floor
    available_range = baseline_mid_masked - blank_floor
    fraction_recovered = gap_to_floor / available_range if available_range > 1e-6 else float("nan")

    print(f"  dropped-mid proxy quality (masked): {dropped_mid_quality:.2f} dB")
    print(f"  blank floor (masked): {blank_floor:.2f} dB")
    print(f"  FFTs saved by dropping mid: {ffts_saved_pct:+.1f}%")
    print(f"  gap to ceiling: {gap_to_ceiling:.2f} dB, gap to floor: {gap_to_floor:.2f} dB")
    print(f"  NORMALIZED fraction of available range recovered for free: {fraction_recovered*100:.1f}%")

    return {"name": name, "baseline_mid": baseline_mid_masked, "proxy": dropped_mid_quality,
            "floor": blank_floor, "ffts_saved_pct": ffts_saved_pct,
            "gap_to_ceiling": gap_to_ceiling, "gap_to_floor": gap_to_floor,
            "fraction_recovered": fraction_recovered,
            "mid_recon": mid_recon.detach().cpu().numpy(), "mid_target": mid,
            "baseline_mid_recon": baseline_mid_recon.detach().cpu().numpy()}


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()

    print("Warming up GPU...")
    _ = multiplane_gs_torch(make_realistic_multiplane_target(SHAPE), DEPTHS_M, WAVELENGTH, DX, 1,
                             device=DEVICE, pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    closeup = make_closeup_scene(SHAPE)
    layered = make_layered_scene(SHAPE)

    r_closeup = evaluate_profile("CLOSEUP (dominant near, sparse mid, flat far)", closeup)
    r_layered = evaluate_profile("LAYERED (street-scene-style, substantial content at all 3 depths)", layered)

    print("\n" + "#" * 90)
    print("COMPARISON (full resolution)")
    print("#" * 90)
    print(f"{'Profile':<12}{'Ceiling':<10}{'Proxy':<10}{'Floor':<10}{'Gap-to-ceiling':<16}{'Gap-to-floor':<14}"
          f"{'Fraction free':<15}{'FFTs saved'}")
    for r in [r_closeup, r_layered]:
        print(f"{r['name'].split()[0]:<12}{r['baseline_mid']:<10.2f}{r['proxy']:<10.2f}{r['floor']:<10.2f}"
              f"{r['gap_to_ceiling']:<16.2f}{r['gap_to_floor']:<14.2f}{r['fraction_recovered']*100:<14.1f}%"
              f"{r['ffts_saved_pct']:+.1f}%")

    print(f"\nCompare against 512x512: CLOSEUP=38.7%, LAYERED=25.8% (normalized fraction recovered)")
    print(f"Full resolution: CLOSEUP={r_closeup['fraction_recovered']*100:.1f}%, "
          f"LAYERED={r_layered['fraction_recovered']*100:.1f}%")

    diff = r_closeup["fraction_recovered"] - r_layered["fraction_recovered"]
    if diff > 0.10:
        print("\nFinding SURVIVES at full resolution: CLOSEUP recovers a meaningfully larger fraction of its")
        print("available range for free than LAYERED does. Content-adaptive plane count is now supported at")
        print("both scales, not just 512x512 -- a real, reproducible physical finding for 10.8.")
    elif abs(diff) <= 0.10:
        print("\nFinding DOES NOT survive at full resolution -- the two profiles now recover similar fractions.")
        print("The 512x512 result was scale-dependent, matching this project's recurring pattern (10.3, etc.)")
        print("Do not carry the small-scale content-adaptivity claim forward without this caveat.")
    else:
        print("\nFinding REVERSES at full resolution -- LAYERED now recovers more than CLOSEUP. Report this")
        print("divergence directly; do not average it with the small-scale result.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, r in enumerate([r_closeup, r_layered]):
        axes[i, 0].imshow(r["mid_target"], cmap="gray")
        axes[i, 0].set_title(f"{r['name'].split()[0]}: real mid content")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(r["mid_recon"], cmap="gray")
        axes[i, 1].set_title(f"2-plane proxy ({r['proxy']:.1f} dB)")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(r["baseline_mid_recon"], cmap="gray")
        axes[i, 2].set_title(f"3-plane actual ({r['baseline_mid']:.1f} dB)")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig("two_plane_dropoff_content_dependence_fullres.png", dpi=130)
    print("\nSaved plot: two_plane_dropoff_content_dependence_fullres.png")


if __name__ == "__main__":
    main()
