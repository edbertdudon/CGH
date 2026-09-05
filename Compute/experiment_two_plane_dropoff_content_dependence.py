"""
Follow-up to experiment_two_plane_dropoff.py, testing a hypothesis
raised directly from that result: is "2 planes sufficient" a fixed
architectural answer, or does it depend on what's actually happening at
each depth? Two content profiles, same methodology (masked PSNR for any
sparse plane, per the fix applied to the original experiment):

  CLOSEUP: near carries a dominant sharp object (icon), mid carries only
    a small sparse accent, far is a near-uniform soft background --
    modeling "looking closely at one thing, little happening at other
    depths" (e.g. a shallow-depth-of-field product shot).

  LAYERED (street-scene-style): near, mid, and far each carry genuinely
    distinct, substantial, non-sparse content -- a foreground object, a
    busy midground (photo-like gradient/texture), and a busy but
    different-textured background (synthetic checker/ring pattern) --
    modeling a scene where real content actually exists at every depth
    simultaneously (foreground / midground / background all "doing
    something").

For each profile: solve 3-plane baseline, solve 2-plane (mid dropped),
propagate the 2-plane phase to the dropped mid depth (the "does content
leak through anyway" proxy), and compare masked PSNR at mid against both
the 3-plane ceiling and a blank-field floor -- identical methodology to
the corrected two_plane_dropoff.py run, applied side by side to both
content profiles instead of just one.

If the leaked-signal-vs-ceiling gap is small for CLOSEUP (little lost by
dropping mid, because there wasn't much there to lose) but large for
LAYERED (real content genuinely lost), that supports content-adaptive
plane count rather than a fixed 2-vs-3 decision.

Small scale first (512x512), matching this project's usual pattern.

Run on your 3060:
    python3 experiment_two_plane_dropoff_content_dependence.py
"""
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, make_multiplane_target, sparse_content_bounds, make_target_plane
from propagation_torch import angular_spectrum_propagate, safe_abs
from retrieval_torch import multiplane_gs_torch, _to_tensor_targets, _psnr_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]     # near, mid, far
MID_INDEX = 1
N_ITERS_BUDGET = 150
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
        device=DEVICE, seed=seed, pad_factor=PAD_FACTOR
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    plateau = find_plateau_robust(history)
    ffts_per_iter = 4 * len(depths_subset)
    return {"phase": phase, "history": history, "plateau": plateau,
            "final": history[-1], "ffts_at_plateau": plateau * ffts_per_iter, "elapsed": elapsed}


def make_closeup_scene(shape):
    """Dominant near object, sparse small accent at mid, near-uniform soft far background."""
    base = make_realistic_multiplane_target(shape)
    near = base[0]  # icon, dominant sharp object

    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    cy, cx = ny * 0.7, nx * 0.7
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mid = (r < min(ny, nx) * 0.03).astype(float)  # tiny sparse accent, far corner

    far = np.full(shape, 0.15) + 0.03 * np.random.default_rng(1).standard_normal(shape)
    far = np.clip(far, 0, 1)  # near-uniform soft background, minimal structure

    return [near, mid, far]


def make_layered_scene(shape):
    """Distinct, substantial content at every depth -- foreground/midground/background."""
    base = make_realistic_multiplane_target(shape)
    near = base[0]     # icon, foreground object
    mid = base[2]      # photo-like gradient/texture scene, busy midground

    far_planes = make_multiplane_target(shape, n_planes=3, soft=True, sigma=8.0)
    far = far_planes[1]  # ring pattern, blurred -- busy, texturally distinct background

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
          f"mid quality (masked)={baseline_mid_masked:.2f} dB")

    two_plane_targets = [near, far]
    two_plane_depths = [DEPTHS_M[0], DEPTHS_M[2]]
    two_plane = solve_multiplane(two_plane_targets, two_plane_depths, seed=0)
    print(f"  2-plane (mid dropped): plateau={two_plane['plateau']}, FFTs/frame={two_plane['ffts_at_plateau']}")

    mid_recon = safe_abs(angular_spectrum_propagate(
        torch.exp(1j * two_plane["phase"].to(torch.float32)), WAVELENGTH, DX, DEPTHS_M[MID_INDEX], pad_factor=PAD_FACTOR
    ))
    dropped_mid_quality = _psnr_torch(mid_recon[rows, cols], targets_amp[MID_INDEX][rows, cols])

    blank_field = torch.zeros_like(targets_amp[MID_INDEX])
    blank_floor = _psnr_torch(blank_field[rows, cols], targets_amp[MID_INDEX][rows, cols])

    ffts_saved_pct = 100 * (1 - two_plane["ffts_at_plateau"] / baseline["ffts_at_plateau"])
    gap_to_ceiling = baseline_mid_masked - dropped_mid_quality
    gap_to_floor = dropped_mid_quality - blank_floor

    # Raw dB gaps aren't comparable across profiles: sparse_content_bounds masks CLOSEUP's
    # mid to a tiny patch but leaves LAYERED's dense mid unmasked (whole 512x512 frame) --
    # different pixel counts and different inherent achievable ceilings (busy content is
    # harder to reconstruct at all, independent of the plane-count question) both distort a
    # direct dB comparison. This ratio is scale-invariant within each profile's own units --
    # it answers "of the possible improvement from floor to ceiling, what fraction does the
    # leaked signal already capture for free," which is comparable across profiles.
    available_range = baseline_mid_masked - blank_floor
    fraction_recovered = gap_to_floor / available_range if available_range > 1e-6 else float("nan")

    print(f"  dropped-mid proxy quality (masked): {dropped_mid_quality:.2f} dB")
    print(f"  blank floor (masked): {blank_floor:.2f} dB")
    print(f"  FFTs saved by dropping mid: {ffts_saved_pct:+.1f}%")
    print(f"  gap to ceiling: {gap_to_ceiling:.2f} dB, gap to floor: {gap_to_floor:.2f} dB")
    print(f"  NORMALIZED fraction of available range (floor->ceiling) recovered for free: "
          f"{fraction_recovered*100:.1f}%")

    return {"name": name, "baseline_mid": baseline_mid_masked, "proxy": dropped_mid_quality,
            "floor": blank_floor, "ffts_saved_pct": ffts_saved_pct,
            "gap_to_ceiling": gap_to_ceiling, "gap_to_floor": gap_to_floor,
            "fraction_recovered": fraction_recovered,
            "mid_recon": mid_recon.detach().cpu().numpy(), "mid_target": mid,
            "baseline_mid_recon": baseline_mid_recon.detach().cpu().numpy()}


def main():
    print(f"Device: {DEVICE}")
    closeup = make_closeup_scene(SHAPE)
    layered = make_layered_scene(SHAPE)

    r_closeup = evaluate_profile("CLOSEUP (dominant near, sparse mid, flat far)", closeup)
    r_layered = evaluate_profile("LAYERED (street-scene-style, substantial content at all 3 depths)", layered)

    print("\n" + "#" * 90)
    print("COMPARISON")
    print("#" * 90)
    print(f"{'Profile':<12}{'Ceiling':<10}{'Proxy':<10}{'Floor':<10}{'Gap-to-ceiling':<16}{'Gap-to-floor':<14}"
          f"{'Fraction free':<15}{'FFTs saved'}")
    for r in [r_closeup, r_layered]:
        print(f"{r['name'].split()[0]:<12}{r['baseline_mid']:<10.2f}{r['proxy']:<10.2f}{r['floor']:<10.2f}"
              f"{r['gap_to_ceiling']:<16.2f}{r['gap_to_floor']:<14.2f}{r['fraction_recovered']*100:<14.1f}%"
              f"{r['ffts_saved_pct']:+.1f}%")

    print(f"\nRaw dB gap-to-ceiling (NOT comparable across profiles -- different measurement region")
    print(f"sizes and different inherent ceilings): CLOSEUP={r_closeup['gap_to_ceiling']:.2f}, "
          f"LAYERED={r_layered['gap_to_ceiling']:.2f} -- this ordering was backwards vs. the hypothesis.")
    print(f"Normalized fraction-of-available-range recovered for free (comparable across profiles): "
          f"CLOSEUP={r_closeup['fraction_recovered']*100:.1f}%, LAYERED={r_layered['fraction_recovered']*100:.1f}%")

    diff = r_closeup["fraction_recovered"] - r_layered["fraction_recovered"]
    if diff > 0.10:
        print("\nOnce normalized, CLOSEUP recovers a meaningfully LARGER fraction of its available range for")
        print("free than LAYERED does -- supports the original hypothesis after all: sparse/simple depth")
        print("distributions leak proportionally more 'free' content when a plane is dropped, while scenes")
        print("with genuinely distinct content at every depth lose proportionally more. The raw dB comparison")
        print("was misleading because of a real measurement confound, not because the hypothesis was wrong.")
    elif abs(diff) <= 0.10:
        print("\nNormalized fractions are close -- does NOT support content-adaptive plane count once the")
        print("measurement-region/ceiling confound is corrected for.")
    else:
        print("\nEven normalized, LAYERED recovers a larger fraction than CLOSEUP -- the hypothesis does not")
        print("hold even after correcting for the measurement confound.")

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
    plt.savefig("two_plane_dropoff_content_dependence.png", dpi=130)
    print("\nSaved plot: two_plane_dropoff_content_dependence.png")


if __name__ == "__main__":
    main()
