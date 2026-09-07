"""
Measures the Joint eyebox-training method's own true FFT-cost-to-converge,
with the same rigor this project's compute baseline has required since the
552->2,618 FFT correction: 8 seeds, a generous iteration budget, real FFT
counting (fft_counter_torch), and find_plateau_robust -- NOT the fixed,
unverified 500-step training budget every prior eyebox test used.

Why this matters now: discrete-tile has been dropped in favor of Joint
(avoids discrete-tile's structural "snapping" between fixed eye-position
tiles). This unlocks a compute win independent of every other lever found
so far: discrete-tile's frame-rate requirement (360Hz = 90Hz base x 4,
Section 4.3) exists specifically because it must cycle 4 separately-
solved patterns fast enough to blend into one image. Joint needs only ONE
solved pattern to already cover the eyebox -- if that holds, the required
frame rate for a Joint-only architecture drops back to the 90Hz base
rate, a 4x cut to required throughput/power on top of every other lever.
But that claim depends on Joint's own true FFTs-to-converge, which this
measures for the first time.

Content and methodology matched to the ALREADY-RECOMMENDED relaxed
resolution (10.10/10.14: confirmed twice, no quality cost, cheaper) --
not the old current resolution being phased out -- so this result is
directly usable for real planning: procedural content (matching the
original baseline's own methodology), 1,422x2,475px, pad_factor=2.0
fixed. The GS/discrete-tile baseline at this exact resolution is already
known (10.10/A.14.9): ~2,697 FFTs/frame (103% of the 2,618 current-res
baseline), 1.67 GFLOP/FFT, ~245W average power for ONE solved pattern.

Eyebox range matches the confirmed wide-range condition (MAX_U_FAR_PX=10,
far-plane-equivalent pixels at DX=2um -- the same physical eyebox width
tested throughout this project's eyebox work, unaffected by resolution
since DX, the physical pixel pitch, is unchanged).

Quality tracked two ways: the training batch's own reconstructions (free,
already computed each step, used to detect the plateau) and a proper
held-out 15-viewpoint evaluation on the FINAL phase pattern (used for the
honestly-reported quality figure -- valid as a plateau-quality proxy
specifically because find_plateau_robust's sustained-hold definition
means the curve stayed flat through to the end of the run whenever
still_rising is False).

Run on your 3060 (calibrate first with --calibrate, expect the full run
to take a while at this resolution):
    python3 experiment_joint_true_baseline.py --calibrate
    python3 experiment_joint_true_baseline.py
"""
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, r"D:\Documents\CGH\Demo")
from fft_counter_torch import counter
from targets import make_realistic_multiplane_target, sparse_content_bounds
from retrieval_torch import _to_tensor_targets
from propagation_torch import angular_spectrum_propagate, safe_abs
from metrics import psnr_intensity
from convergence import find_plateau_robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (1422, 2475)  # the RECOMMENDED relaxed resolution, not the old current one
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
Z_FAR = DEPTHS_M[-1]
PARALLAX_RATIOS = [Z_FAR / z for z in DEPTHS_M]

MAX_U_FAR_PX = 10.0
BATCH_SIZE = 4
LR = 0.02
SEEDS = list(range(8))
N_STEPS_MAX = 800  # revisit after --calibrate timing

N_TEST_VIEWPOINTS = 15

TOPS_PER_W = 6.5
POWER_BUDGET_MW = 500
BASE_HZ = 90  # perceptual base rate, no discrete-tile 4x cycling needed for Joint

# already-established GS/discrete-tile baseline at this exact resolution (10.10/A.14.9)
GS_BASELINE_FFTS_PER_FRAME = 2697
GS_BASELINE_GFLOP_PER_FFT = 1.67
GS_BASELINE_POWER_MW = 245000  # ~245W, one solved pattern, at 360Hz discrete-tile cycling


def pad_crop_shift_subpixel(plane, dx_px, dy_px=0.0):
    t = torch.as_tensor(plane)
    h, w = t.shape
    dx_i, dy_i = int(round(dx_px)), int(round(dy_px))
    shifted = torch.zeros_like(t)
    src_x0, src_x1 = max(0, -dx_i), w - max(0, dx_i)
    dst_x0, dst_x1 = max(0, dx_i), w - max(0, -dx_i)
    src_y0, src_y1 = max(0, -dy_i), h - max(0, dy_i)
    dst_y0, dst_y1 = max(0, dy_i), h - max(0, -dy_i)
    if src_x1 > src_x0 and src_y1 > src_y0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = t[src_y0:src_y1, src_x0:src_x1]
    return shifted.numpy()


def viewpoint_targets(base_planes, u_far_px):
    return [pad_crop_shift_subpixel(p, u_far_px * ratio) for p, ratio in zip(base_planes, PARALLAX_RATIOS)]


def masked_quality(recon_planes, targets):
    """Matches the ORIGINAL discrete-tile/GS baseline's own masking convention
    (sparse_content_bounds for near/mid, whole-frame for far) -- procedural
    content only, so the bounding-box mask is valid here (that bug was
    specific to real, scattered photo content, not this project's single-
    blob procedural set). Kept identical to the GS baseline for a fair
    apples-to-apples comparison."""
    psnrs = []
    for i, (r, t) in enumerate(zip(recon_planes, targets)):
        if i in (0, 1):
            rows, cols = sparse_content_bounds(t)
            psnrs.append(psnr_intensity(r[rows, cols], t[rows, cols]))
        else:
            psnrs.append(psnr_intensity(r, t))
    return psnrs


def propagate_all_planes(slm_phase):
    field = torch.exp(1j * slm_phase)
    return [safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR)) for z in DEPTHS_M]


def train_joint(base_planes, seed, n_steps):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    slm_phase = (torch.rand(SHAPE, device=DEVICE) * 2 * np.pi - np.pi).requires_grad_(True)
    opt = torch.optim.Adam([slm_phase], lr=LR)

    history = []
    for step in range(n_steps):
        us = rng.uniform(-MAX_U_FAR_PX, MAX_U_FAR_PX, size=BATCH_SIZE)
        opt.zero_grad()
        field = torch.exp(1j * slm_phase)
        loss = torch.tensor(0.0, device=DEVICE)
        batch_quals = []
        for u in us:
            targets_np = viewpoint_targets(base_planes, u)
            targets_amp = _to_tensor_targets(targets_np, DEVICE)
            recon_planes = []
            for target_amp, z in zip(targets_amp, DEPTHS_M):
                recon = safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR))
                loss = loss + torch.mean((recon - target_amp) ** 2) / BATCH_SIZE
                recon_planes.append(recon.detach().cpu().numpy())
            batch_quals.append(sum(masked_quality(recon_planes, targets_np)) / 3)
        history.append(sum(batch_quals) / len(batch_quals))
        loss.backward()
        opt.step()

    return slm_phase.detach(), history


def held_out_eval(base_planes, slm_phase):
    test_viewpoints = list(np.linspace(-MAX_U_FAR_PX, MAX_U_FAR_PX, N_TEST_VIEWPOINTS))
    recon_all = [r.detach().cpu().numpy() for r in propagate_all_planes(slm_phase)]
    vals = []
    for u in test_viewpoints:
        targets_np = viewpoint_targets(base_planes, u)
        vals.append(sum(masked_quality(recon_all, targets_np)) / 3)
    return float(np.mean(vals))


def calibrate():
    print(f"Device: {DEVICE}, SHAPE={SHAPE}")
    base_planes = make_realistic_multiplane_target(SHAPE)
    for n in (10, 30):
        t0 = time.time()
        _, hist = train_joint(base_planes, seed=0, n_steps=n)
        dt = time.time() - t0
        print(f"   {n} steps: {dt:.1f}s ({dt/n:.2f}s/step) -> full 8-seed x {N_STEPS_MAX}-step run "
              f"would take ~{dt/n*N_STEPS_MAX*8/60:.0f} min")


def main():
    print(f"Device: {DEVICE}, SHAPE={SHAPE}, N_STEPS_MAX={N_STEPS_MAX}, seeds={SEEDS}")
    t_start = time.time()
    base_planes = make_realistic_multiplane_target(SHAPE)

    results = []
    for seed in SEEDS:
        counter.reset()
        t0 = time.time()
        slm_phase, history = train_joint(base_planes, seed, N_STEPS_MAX)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        total_ffts = counter.count
        ffts_per_step = total_ffts / N_STEPS_MAX

        plateau_iter, reference, still_rising = find_plateau_robust(history)
        ffts_at_plateau = ffts_per_step * plateau_iter
        gflop_per_fft = 5 * (SHAPE[0] * PAD_FACTOR * SHAPE[1] * PAD_FACTOR) * np.log2(SHAPE[0] * PAD_FACTOR * SHAPE[1] * PAD_FACTOR) / 1e9
        gflop_per_frame = ffts_at_plateau * gflop_per_fft
        tflops_needed_90hz = gflop_per_frame * BASE_HZ / 1000
        power_mw_90hz = tflops_needed_90hz / TOPS_PER_W * 1000

        held_out_quality = held_out_eval(base_planes, slm_phase)

        results.append({
            "seed": seed, "plateau_iter": plateau_iter, "still_rising": still_rising,
            "ffts_at_plateau": ffts_at_plateau, "power_mw_90hz": power_mw_90hz,
            "batch_quality_at_plateau": reference, "held_out_quality": held_out_quality,
            "elapsed": elapsed,
        })
        print(f"   seed={seed}: plateau@{plateau_iter}/{N_STEPS_MAX}, batch-quality={reference:.2f}dB, "
              f"held-out quality={held_out_quality:.2f}dB, {ffts_at_plateau:.0f} FFTs/frame, "
              f"{power_mw_90hz:.0f}mW @ 90Hz ({power_mw_90hz/POWER_BUDGET_MW:.1f}x budget), "
              f"{elapsed:.1f}s{' STILL RISING' if still_rising else ''}")

    ffts = [r["ffts_at_plateau"] for r in results]
    powers = [r["power_mw_90hz"] for r in results]
    quals = [r["held_out_quality"] for r in results]
    print(f"\n{'='*100}")
    print(f"Joint @ relaxed resolution, {BASE_HZ}Hz (no discrete-tile cycling needed): "
          f"{np.mean(ffts):.0f} FFTs/frame avg (range {min(ffts):.0f}-{max(ffts):.0f}), "
          f"{np.mean(powers):.0f}mW avg ({np.mean(powers)/POWER_BUDGET_MW:.1f}x budget), "
          f"held-out quality {np.mean(quals):.2f}dB (range {min(quals):.2f}-{max(quals):.2f})")
    print(f"GS/discrete-tile baseline, same resolution, ONE pattern: {GS_BASELINE_FFTS_PER_FRAME} FFTs/frame, "
          f"{GS_BASELINE_POWER_MW/1000:.0f}mW @ 90Hz-equivalent (x4 needed for real discrete-tile use -> "
          f"{GS_BASELINE_POWER_MW*4/1000:.0f}mW @ 360Hz, {GS_BASELINE_POWER_MW*4/POWER_BUDGET_MW:.0f}x budget)")

    any_rising = any(r["still_rising"] for r in results)
    if any_rising:
        print("\nWARNING: at least one seed was still rising at N_STEPS_MAX -- extend the budget before trusting this fully.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate()
    else:
        main()
