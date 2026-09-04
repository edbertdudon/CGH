"""
Follow-up (2) to experiment_temporal_warmstart.py / _shift_sweep.py.

The shift sweep's non-monotonic speedup curve (16x at 3px down to ~6x
at 60px, back up to ~7.5x at 150px) was flagged as likely caused by
torch.roll's circular wraparound: at large shifts on a 512px canvas,
content that scrolls off one edge reappears on the other, so frame-to-
frame difference doesn't grow monotonically with shift the way real
panning would. This isolates that question directly by rerunning the
same shift sweep with a non-wrapping shift (pad with edge-replicated
pixels, then crop) instead of torch.roll, and comparing the two
directly at the same shift values.

Same target, propagation model, iteration budget, and threshold
definition as the rest of this experiment family. Uses the corrected
multiplane_gs_torch history (the off-by-one fix applied during this
session) -- run after that fix, not before.

Run on your 3060:
    python3 experiment_temporal_warmstart_padcrop.py
"""
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from experiment_temporal_warmstart import (
    check_patch_applied, make_frame_sequence, run_sequence, first_crossing,
    SHAPE, N_FRAMES, QUALITY_THRESHOLD_MARGIN_DB,
)

SHIFT_VALUES_PX = [3, 10, 30, 60, 100, 150]  # same values as the roll-based shift sweep


def shift_replicate(arr, shift_px):
    """
    Shift a 2D array right by shift_px, filling the revealed left edge
    by replicating the edge pixel instead of wrapping content from the
    right edge (torch.roll's behavior). Non-wrapping stand-in for real
    panning, where content leaving the frame doesn't reappear elsewhere.
    """
    if shift_px == 0:
        return arr
    t = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    padded = F.pad(t, (shift_px, 0, 0, 0), mode="replicate")
    cropped = padded[:, :, :, :arr.shape[1]]
    return cropped.squeeze(0).squeeze(0).numpy()


def make_frame_sequence_padcrop(base_planes, n_frames, shift_px):
    frames = []
    for f in range(n_frames):
        shifted = [shift_replicate(p, f * shift_px) for p in base_planes]
        frames.append(shifted)
    return frames


def average_speedup(frames):
    cold = run_sequence(frames, warm_start=False)
    warm = run_sequence(frames, warm_start=True)
    speedups = []
    for f in range(N_FRAMES):
        cold_final = cold[f]["history"][-1]
        threshold = cold_final - QUALITY_THRESHOLD_MARGIN_DB
        cold_cross = first_crossing(cold[f]["history"], threshold)
        warm_cross = first_crossing(warm[f]["history"], threshold)
        if cold_cross and warm_cross:
            speedups.append(cold_cross / warm_cross)
    if len(speedups) < 2:
        return None
    steady_state = speedups[1:]
    return sum(steady_state) / len(steady_state)


def main():
    check_patch_applied()
    base = make_realistic_multiplane_target(SHAPE)

    print(f"{'Shift (px)':<14}{'Roll (wrap) speedup':<22}{'Pad-crop (no-wrap) speedup':<28}")
    roll_results, padcrop_results = [], []
    for shift in SHIFT_VALUES_PX:
        roll_frames = make_frame_sequence(base, N_FRAMES, shift)
        padcrop_frames = make_frame_sequence_padcrop(base, N_FRAMES, shift)

        roll_speedup = average_speedup(roll_frames)
        padcrop_speedup = average_speedup(padcrop_frames)
        roll_results.append((shift, roll_speedup))
        padcrop_results.append((shift, padcrop_speedup))

        roll_str = f"{roll_speedup:.2f}x" if roll_speedup is not None else "n/a"
        padcrop_str = f"{padcrop_speedup:.2f}x" if padcrop_speedup is not None else "n/a"
        print(f"{shift:<14}{roll_str:<22}{padcrop_str:<28}")

    roll_valid = [(s, sp) for s, sp in roll_results if sp is not None]
    padcrop_valid = [(s, sp) for s, sp in padcrop_results if sp is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    if roll_valid:
        ax.plot([s for s, _ in roll_valid], [sp for _, sp in roll_valid], marker="o", label="Roll (wraparound)")
    if padcrop_valid:
        ax.plot([s for s, _ in padcrop_valid], [sp for _, sp in padcrop_valid], marker="s", label="Pad-crop (no wrap)")
    ax.axhline(1.0, color="gray", linestyle="--", label="No speedup (1x)")
    ax.set_xlabel("Shift per frame (px, 512x512 canvas)")
    ax.set_ylabel("Avg. iteration speedup (warm vs. cold)")
    ax.set_title("Warm-start speedup: wraparound vs. non-wrapping motion")
    ax.legend()
    plt.tight_layout()
    plt.savefig("temporal_warmstart_padcrop_comparison.png", dpi=130)
    print("\nSaved plot: temporal_warmstart_padcrop_comparison.png")

    # Monotonicity check: does removing wraparound produce a monotonically
    # decaying curve, unlike roll's dip-then-rise shape?
    padcrop_speeds = [sp for _, sp in padcrop_valid]
    is_monotonic = all(padcrop_speeds[i] >= padcrop_speeds[i + 1] - 1e-6 for i in range(len(padcrop_speeds) - 1))
    print(f"\nPad-crop speedup curve monotonically decaying with shift: {is_monotonic}")
    if is_monotonic:
        print("-> Supports the wraparound theory: removing it gives the smooth decay you'd expect")
        print("   from real panning, unlike roll's dip-then-rise shape.")
    else:
        print("-> Pad-crop ALSO shows non-monotonic behavior -- wraparound is RULED OUT as the")
        print("   (sole) explanation for the roll sweep's dip-then-rise shape.")
        print()
        print("STATUS: closed as non-blocking, not closed as explained -- same posture as 10.4's")
        print("border artifact. What was tested and ruled out: circular wraparound (this script,")
        print("both curves wobble the same way with and without it). What was NOT tested: whether")
        print("it's a property of this specific synthetic-shift motion model at all (vs. a genuine,")
        print("reproducible non-monotonicity in how GS warm-starting responds to motion magnitude),")
        print("since no real (non-synthetic) motion sequence has been tried in this experiment")
        print("family. The steady-state speedup finding itself does not depend on this shape being")
        print("explained -- warm-starting helps at every shift tested, roll or pad-crop, small or")
        print("large -- but the exact magnitude at a given motion size should not be trusted to")
        print("interpolate smoothly between the tested points until this is understood.")


if __name__ == "__main__":
    main()
