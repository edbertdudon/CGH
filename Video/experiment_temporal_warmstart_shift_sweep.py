"""
Follow-up to experiment_temporal_warmstart.py.

That experiment measured a 15.71x average warm-start speedup at
SHIFT_PX_PER_FRAME=3 on a 512x512 canvas -- but warm-start crossed the
quality threshold at iteration 1 on every frame after the first, which
is consistent with the synthetic motion being too gentle to be a fair
test (a 3px shift on 512px, via torch.roll, barely changes the frame
at all). This reruns the same cold-vs-warm comparison across a range
of shift magnitudes to see whether the speedup survives motion large
enough to actually change frame content, or collapses toward 1x as
shift grows -- the same "does it survive scale/realism" check applied
throughout Section 10 (see 10.3's resolution sweep).

Reuses every function from experiment_temporal_warmstart.py unchanged
-- same target, same propagation model, same iteration budget, same
threshold definition. Only SHIFT_PX_PER_FRAME varies.

Run on your 3060:
    python3 experiment_temporal_warmstart_shift_sweep.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from experiment_temporal_warmstart import (
    check_patch_applied, make_frame_sequence, run_sequence, first_crossing,
    SHAPE, N_FRAMES, QUALITY_THRESHOLD_MARGIN_DB,
)

SHIFT_VALUES_PX = [3, 10, 30, 60, 100, 150]


def average_speedup(frames):
    cold = run_sequence(frames, warm_start=False)
    warm = run_sequence(frames, warm_start=True)

    speedups = []
    warm_at_iter1 = 0
    for f in range(N_FRAMES):
        cold_final = cold[f]["history"][-1]
        threshold = cold_final - QUALITY_THRESHOLD_MARGIN_DB
        cold_cross = first_crossing(cold[f]["history"], threshold)
        warm_cross = first_crossing(warm[f]["history"], threshold)
        if cold_cross and warm_cross:
            speedups.append(cold_cross / warm_cross)
            if f > 0 and warm_cross == 1:
                warm_at_iter1 += 1

    if len(speedups) < 2:
        return None, warm_at_iter1
    steady_state = speedups[1:]
    return sum(steady_state) / len(steady_state), warm_at_iter1


def main():
    check_patch_applied()
    base = make_realistic_multiplane_target(SHAPE)

    print(f"{'Shift (px)':<14}{'Shift (% of frame)':<22}{'Avg speedup':<14}{'Frames warm=iter1':<20}")
    results = []
    for shift in SHIFT_VALUES_PX:
        frames = make_frame_sequence(base, N_FRAMES, shift)
        avg_speedup, warm_at_iter1 = average_speedup(frames)
        pct = 100 * shift / SHAPE[1]
        speedup_str = f"{avg_speedup:.2f}x" if avg_speedup is not None else "n/a"
        print(f"{shift:<14}{pct:<22.1f}{speedup_str:<14}{warm_at_iter1}/{N_FRAMES - 1:<18}")
        results.append((shift, avg_speedup))

    valid = [(s, sp) for s, sp in results if sp is not None]
    if valid:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([s for s, _ in valid], [sp for _, sp in valid], marker="o")
        ax.axhline(1.0, color="gray", linestyle="--", label="No speedup (1x)")
        ax.set_xlabel("Shift per frame (px, 512x512 canvas)")
        ax.set_ylabel("Avg. iteration speedup (warm vs. cold)")
        ax.set_title("Warm-start speedup vs. synthetic motion magnitude")
        ax.legend()
        plt.tight_layout()
        plt.savefig("temporal_warmstart_shift_sweep.png", dpi=130)
        print("\nSaved plot: temporal_warmstart_shift_sweep.png")

    if valid and valid[-1][1] is not None and valid[-1][1] < 1.3:
        print("\nSpeedup collapses toward 1x at large shift -- consistent with the iter-1 crossing")
        print("at small shift being an artifact of near-static synthetic motion, not a real")
        print("warm-start benefit. Do not update Gap 2's compute figure on this evidence.")
    elif valid and all(sp > 1.3 for _, sp in valid):
        print("\nSpeedup holds even at large shift -- real evidence warm-starting helps beyond")
        print("near-static content. Worth a full-resolution, real-content rerun before touching")
        print("Gap 2's compute figure.")
    else:
        print("\nMixed: speedup present at small shift, degrades at large shift -- worth reporting")
        print("as a content-motion-dependent effect, not a flat multiplier, if this goes in Section 10.")


if __name__ == "__main__":
    main()
