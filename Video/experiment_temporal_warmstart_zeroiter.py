"""
Follow-up (1) to experiment_temporal_warmstart.py / _shift_sweep.py.

The shift sweep showed warm-start crossing the quality threshold at
iteration 1 for most frames at small shift, decaying but never
vanishing as shift grows. That conflates two different effects:

  (a) "free" quality from just REUSING the previous frame's phase
      unmodified, zero GS iterations -- pure phase similarity between
      consecutive frames.
  (b) quality gained from actually running GS iterations starting from
      that reused phase, i.e. the warm-start optimization doing real
      work on top of (a).

This measures (a) directly -- propagate the previous frame's converged
phase through the CURRENT frame's target with no correction applied at
all -- and reports it alongside warm-start's 1-iteration and
cold-start's final quality, so the two effects can be told apart
instead of both hiding inside a single "iterations to threshold"
number.

Reuses the same target, propagation model, and threshold definition as
experiment_temporal_warmstart.py. Same small-scale-first discipline.

Run on your 3060:
    python3 experiment_temporal_warmstart_zeroiter.py
"""
import torch

from targets import make_realistic_multiplane_target
from retrieval_torch import _to_tensor_targets, _psnr_torch
from propagation_torch import angular_spectrum_propagate, safe_abs
from experiment_temporal_warmstart import (
    check_patch_applied, make_frame_sequence, run_sequence,
    SHAPE, DEVICE, WAVELENGTH, DX, PAD_FACTOR, DEPTHS_M, N_FRAMES,
    QUALITY_THRESHOLD_MARGIN_DB,
)

SHIFT_VALUES_PX = [3, 30, 60]  # spans the range where the shift sweep showed the effect shrinking


def zero_iteration_quality(phase, target_planes):
    """PSNR of applying `phase` directly to `target_planes`, no GS correction at all."""
    targets_amp = _to_tensor_targets(target_planes, DEVICE)
    slm_field = torch.exp(1j * phase.to(device=DEVICE, dtype=torch.float32))
    psnrs = []
    for target_amp, z in zip(targets_amp, DEPTHS_M):
        field_at_plane = angular_spectrum_propagate(slm_field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR)
        recon_amp = safe_abs(field_at_plane)
        psnrs.append(_psnr_torch(recon_amp, target_amp))
    return sum(psnrs) / len(psnrs)


def main():
    check_patch_applied()
    base = make_realistic_multiplane_target(SHAPE)

    for shift in SHIFT_VALUES_PX:
        frames = make_frame_sequence(base, N_FRAMES, shift)
        cold = run_sequence(frames, warm_start=False)
        warm = run_sequence(frames, warm_start=True)

        print(f"\n=== shift={shift}px ({100*shift/SHAPE[1]:.1f}% of frame) ===")
        print(f"{'Frame':<8}{'Cold final':<14}{'Zero-iter reuse':<18}{'Warm iter-1':<14}{'Threshold':<12}")
        for f in range(1, N_FRAMES):  # frame 0 has no predecessor to reuse from
            cold_final = cold[f]["history"][-1]
            threshold = cold_final - QUALITY_THRESHOLD_MARGIN_DB
            prev_phase = warm[f - 1]["phase"]
            zero_q = zero_iteration_quality(prev_phase, frames[f])
            warm_iter1 = warm[f]["history"][0]
            print(f"{f:<8}{cold_final:<14.2f}{zero_q:<18.2f}{warm_iter1:<14.2f}{threshold:<12.2f}")

        zero_qs = [
            zero_iteration_quality(warm[f - 1]["phase"], frames[f]) for f in range(1, N_FRAMES)
        ]
        thresholds = [cold[f]["history"][-1] - QUALITY_THRESHOLD_MARGIN_DB for f in range(1, N_FRAMES)]
        already_past = sum(1 for zq, th in zip(zero_qs, thresholds) if zq >= th)
        print(f"Frames where zero-iteration reuse ALONE already clears the threshold: "
              f"{already_past}/{N_FRAMES - 1}")
        if already_past == N_FRAMES - 1:
            print("-> The 'iteration 1' crossings at this shift are fully explained by reuse alone --")
            print("   GS isn't doing any real work here, the phases were already good enough.")
        elif already_past == 0:
            print("-> Reuse alone never clears the threshold here -- the 1-iteration crossings (if any)")
            print("   reflect real GS work on top of a reused starting point, not pure reuse.")
        else:
            print("-> Mixed: reuse alone accounts for some but not all of the observed speedup here.")


if __name__ == "__main__":
    main()
