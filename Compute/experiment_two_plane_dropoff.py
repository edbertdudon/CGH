"""
10.8 flags "are 2 depth planes perceptually sufficient instead of 3" as
a perceptual question, not a simulation one -- correctly. This script
does NOT attempt to answer that. What it measures instead is the
groundwork that question needs: the compute/quality tradeoff of dropping
the middle plane, and a genuinely physical (not perceptual) proxy --
when the GS solve stops being told to reproduce content at the middle
depth, does the resulting phase pattern still produce something close to
that content there anyway? A phase-only hologram isn't blind at
unsolved depths; propagating it anywhere still produces SOME diffraction
pattern. This asks how close that leftover pattern is to the real thing,
which is useful, physically-grounded evidence -- not a verdict on
whether it would look acceptable to a person.

Three conditions, same near/mid/far content throughout:
  A) 3-plane baseline: GS solved against all three depths (near, mid, far)
     -- the existing default, sanity-checked against Gap 7's known
     plane-count quality pattern.
  B) 2-plane: GS solved against only near and far -- mid dropped entirely
     from the optimization target.
  C) B's resulting phase, PROPAGATED to the dropped mid depth and
     compared against the real mid content -- the "does it show up
     anyway" proxy. Compared against both A's actual mid-plane quality
     (the ceiling if you DO solve for it) and a blank-field baseline (the
     floor if dropping mid loses that content entirely).

Uses the final-value-anchored plateau detector (per the standing rule
established after 10.5/A.11's local-window detector produced false
positives) -- implemented here directly rather than assuming access to
the project's internal convergence module, since its exact API isn't
available in this environment. Swap in the project's own validated
detector if preferred; the logic is equivalent (a candidate plateau must
hold within tolerance all the way to the run's true final value, not
just look locally flat).

Small scale first (512x512), matching this project's usual pattern.

Run on your 3060:
    python3 experiment_two_plane_dropoff.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target, sparse_content_bounds
from propagation_torch import angular_spectrum_propagate, safe_abs
from retrieval_torch import multiplane_gs_torch, _to_tensor_targets, _psnr_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]     # near, mid, far
MID_INDEX = 1
N_ITERS_BUDGET = 150                     # generous; final-value-anchored detector needs room to confirm true settling
PLATEAU_EPS_DB = 0.1
PLANE_NAMES = ["near", "mid", "far"]


def find_plateau_robust(history, eps=PLATEAU_EPS_DB):
    """
    Final-value-anchored: returns the first iteration after which EVERY
    subsequent value stays within eps of the run's true final value --
    not just locally flat nearby. This is the corrected-methodology
    detector adopted after 10.5/A.11's local-window detector was found
    to produce false positives (reporting plateau at its own hold-window
    floor regardless of whether the curve had actually settled).
    """
    final = history[-1]
    for i in range(len(history)):
        if all(abs(v - final) < eps for v in history[i:]):
            return i + 1
    return len(history)


def solve_multiplane(targets_subset, depths_subset, seed=0):
    counter.reset()
    t0 = time.time()
    phase, history, _ = multiplane_gs_torch(
        targets_subset, depths_subset, WAVELENGTH, DX, N_ITERS_BUDGET,
        device=DEVICE, seed=seed, pad_factor=PAD_FACTOR
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    plateau = find_plateau_robust(history)
    ffts_per_iter = 4 * len(depths_subset)  # 4 FFT ops per plane per GS iteration (10.1)
    return {"phase": phase, "history": history, "plateau": plateau,
            "final": history[-1], "ffts_at_plateau": plateau * ffts_per_iter, "elapsed": elapsed}


def per_plane_quality(phase, targets_amp, depths):
    """Propagates a solved phase to each given depth and reports PSNR
    against that depth's real target -- used both for the 3-plane
    baseline's per-plane breakdown and for the 2-plane condition's
    "what shows up at the dropped depth" proxy."""
    results = []
    for target_amp, z in zip(targets_amp, depths):
        field = torch.exp(1j * phase.to(torch.float32))
        recon = safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR))
        results.append(_psnr_torch(recon, target_amp))
    return results


def main():
    print(f"Device: {DEVICE}")
    targets = make_realistic_multiplane_target(SHAPE)
    targets_amp = _to_tensor_targets(targets, DEVICE)
    near, mid, far = targets

    print("\nA) 3-plane baseline (near, mid, far)...")
    baseline = solve_multiplane(targets, DEPTHS_M, seed=0)
    baseline_per_plane = per_plane_quality(baseline["phase"], targets_amp, DEPTHS_M)
    print(f"   plateau={baseline['plateau']}, FFTs/frame={baseline['ffts_at_plateau']}")
    print(f"   per-plane: near={baseline_per_plane[0]:.2f} mid={baseline_per_plane[1]:.2f} far={baseline_per_plane[2]:.2f} dB")

    print("\nB) 2-plane (near, far only -- mid dropped from optimization)...")
    two_plane_targets = [near, far]
    two_plane_depths = [DEPTHS_M[0], DEPTHS_M[2]]
    two_plane = solve_multiplane(two_plane_targets, two_plane_depths, seed=0)
    two_plane_amp = _to_tensor_targets(two_plane_targets, DEVICE)
    two_plane_per_plane = per_plane_quality(two_plane["phase"], two_plane_amp, two_plane_depths)
    print(f"   plateau={two_plane['plateau']}, FFTs/frame={two_plane['ffts_at_plateau']}")
    print(f"   per-plane: near={two_plane_per_plane[0]:.2f} far={two_plane_per_plane[1]:.2f} dB")

    print("\nC) Proxy: propagate B's phase to the DROPPED mid depth, compare against real mid content...")
    mid_recon = safe_abs(angular_spectrum_propagate(
        torch.exp(1j * two_plane["phase"].to(torch.float32)), WAVELENGTH, DX, DEPTHS_M[MID_INDEX], pad_factor=PAD_FACTOR
    ))

    # Mid content is TEXT -- sparse, mostly-empty-background content. Section 10.6 already
    # found whole-frame PSNR on sparse content is dominated by trivially-correct background
    # pixels and can badly mislead a comparison (confirmed directly here: the unmasked
    # numbers below initially showed the proxy scoring BELOW a blank field, which is
    # physically backwards -- masking to the actual content region reverses that).
    # sparse_content_bounds expects a numpy array; mid is one already.
    rows, cols = sparse_content_bounds(mid)

    dropped_mid_quality_whole = _psnr_torch(mid_recon, targets_amp[MID_INDEX])
    dropped_mid_quality = _psnr_torch(mid_recon[rows, cols], targets_amp[MID_INDEX][rows, cols])

    baseline_mid_recon_for_mask = safe_abs(angular_spectrum_propagate(
        torch.exp(1j * baseline["phase"].to(torch.float32)), WAVELENGTH, DX, DEPTHS_M[MID_INDEX], pad_factor=PAD_FACTOR
    ))
    baseline_mid_masked = _psnr_torch(baseline_mid_recon_for_mask[rows, cols], targets_amp[MID_INDEX][rows, cols])

    blank_field = torch.zeros_like(targets_amp[MID_INDEX])
    blank_floor_whole = _psnr_torch(blank_field, targets_amp[MID_INDEX])
    blank_floor = _psnr_torch(blank_field[rows, cols], targets_amp[MID_INDEX][rows, cols])

    print(f"   [WHOLE-FRAME, misleading for sparse text content -- kept only for comparison]")
    print(f"     proxy={dropped_mid_quality_whole:.2f} dB, blank floor={blank_floor_whole:.2f} dB")
    print(f"   [MASKED to the actual text content region -- the trustworthy numbers]")
    print(f"     quality at dropped mid depth (2-plane solve, unsolicited): {dropped_mid_quality:.2f} dB")
    print(f"     3-plane baseline's actual solved mid quality (masked): {baseline_mid_masked:.2f} dB")
    print(f"     blank-field floor at mid depth (masked): {blank_floor:.2f} dB")

    print("\n" + "=" * 78)
    print(f"{'Condition':<28}{'FFTs/frame':<14}{'Near':<8}{'Mid(masked)':<12}{'Far':<8}")
    print(f"{'3-plane baseline':<28}{baseline['ffts_at_plateau']:<14}{baseline_per_plane[0]:<8.2f}"
          f"{baseline_mid_masked:<12.2f}{baseline_per_plane[2]:<8.2f}")
    print(f"{'2-plane (mid dropped)':<28}{two_plane['ffts_at_plateau']:<14}{two_plane_per_plane[0]:<8.2f}"
          f"{'--':<12}{two_plane_per_plane[1]:<8.2f}")
    print(f"{'2-plane, mid via proxy':<28}{'(same as above)':<14}{'':<8}{dropped_mid_quality:<12.2f}{'':<8}")
    print("=" * 78)

    ffts_saved_pct = 100 * (1 - two_plane["ffts_at_plateau"] / baseline["ffts_at_plateau"])
    print(f"\nCompute change: {ffts_saved_pct:+.1f}% FFTs/frame vs. 3-plane baseline.")
    print(f"(Whole-frame numbers, kept for the record: proxy={dropped_mid_quality_whole:.2f} dB, "
          f"blank floor={blank_floor_whole:.2f} dB -- these show the proxy scoring BELOW blank, which is")
    print(f"physically backwards and is the tell that sparse-content whole-frame PSNR is misleading here,")
    print(f"same issue Section 10.6 already documented. Masked numbers below are the ones to trust.)")
    gap_to_ceiling = baseline_mid_masked - dropped_mid_quality
    gap_to_floor = dropped_mid_quality - blank_floor
    print(f"Dropped-mid quality sits {gap_to_ceiling:.2f} dB below the 3-plane ceiling and "
          f"{gap_to_floor:.2f} dB above the blank floor.")
    if gap_to_floor < 1.0:
        print("Close to the blank floor -- dropping mid essentially loses that content; no free lunch from")
        print("near/far's diffraction pattern. A 2-plane design would need to actually relocate mid's content")
        print("to near or far, not just omit it.")
    elif gap_to_ceiling < 2.0:
        print("Surprisingly close to the 3-plane ceiling -- near/far's own diffraction may be doing meaningful")
        print("work at the mid depth for free. Worth flagging to an optical engineer as a real candidate,")
        print("alongside the actual depth-of-focus question only a person or a proper eye model can answer.")
    else:
        print("Partial: better than blank, well short of solving for it directly. This is exactly the kind of")
        print("number worth handing to whoever does the perceptual/optical judgment call in 10.8 -- not a")
        print("verdict on its own.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mid, cmap="gray"); axes[0].set_title("Real mid content"); axes[0].axis("off")
    axes[1].imshow(mid_recon.detach().cpu().numpy(), cmap="gray")
    axes[1].set_title(f"2-plane proxy at mid depth ({dropped_mid_quality:.1f} dB)"); axes[1].axis("off")
    baseline_mid_recon = safe_abs(angular_spectrum_propagate(
        torch.exp(1j * baseline["phase"].to(torch.float32)), WAVELENGTH, DX, DEPTHS_M[MID_INDEX], pad_factor=PAD_FACTOR
    )).detach().cpu().numpy()
    axes[2].imshow(baseline_mid_recon, cmap="gray")
    axes[2].set_title(f"3-plane actual mid solve ({baseline_per_plane[1]:.1f} dB)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("two_plane_dropoff_comparison.png", dpi=130)
    print("\nSaved plot: two_plane_dropoff_comparison.png")


if __name__ == "__main__":
    main()
