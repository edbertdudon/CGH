"""
Follow-up (3) to experiment_temporal_warmstart.py.

Every prior test in this family used one continuous scene, panning by a
constant per-frame shift. Real content isn't always like that: a scene
cut (camera cut, app switch, content swap) hands GS a completely
different target with no relationship to the previous frame's converged
phase. If warm-starting can ever be WORSE than cold-starting under that
condition (negative transfer), the architecture can't just provision
for the steady-state speedup measured elsewhere in this family -- it
has to provision for the worst case, which is what actually sizes a
frame-rate/iteration budget in hardware.

Sequence: 3 frames of slow real-content panning (shift=3px, matches the
gentlest case in the shift sweep), then one scene-cut frame swapped to
an unrelated synthetic target (disc/ring/checker, from
targets.make_multiplane_target -- a different content family entirely,
not just a bigger shift of the same scene), then 3 more panning frames
of the ORIGINAL scene (testing whether warm-start recovers cleanly the
frame immediately after a cut, using the cut frame's converged phase as
its seed).

Same propagation model, iteration budget, and threshold definition as
the rest of this family. Uses the corrected multiplane_gs_torch history
(off-by-one fix applied earlier this session).

Run on your 3060:
    python3 experiment_temporal_warmstart_scenecut.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, make_multiplane_target
from experiment_temporal_warmstart import (
    check_patch_applied, make_frame_sequence, run_sequence, first_crossing,
    SHAPE, N_ITERS_BUDGET, QUALITY_THRESHOLD_MARGIN_DB,
)

SHIFT_PX = 3
CUT_FRAME_INDEX = 3       # 0,1,2 = pan; 3 = cut; 4,5,6 = pan again (back to original scene, continuing its own shift schedule)
N_PAN_BEFORE = 3
N_PAN_AFTER = 3


def build_sequence():
    """
    Frames 0..2: realistic scene, panned by SHIFT_PX per frame (as in the
    rest of this family).
    Frame 3: scene cut -- unrelated synthetic disc/ring/checker content,
    no shift relationship to the realistic scene at all.
    Frames 4..6: back to the realistic scene, continuing its own pan
    schedule as if the cut were a momentary interruption (frame 4 uses
    the same content frame 3-of-the-realistic-sequence would have, i.e.
    shift = 3 * SHIFT_PX, etc.) -- tests recovery after the cut, not a
    second unrelated scene.
    """
    base = make_realistic_multiplane_target(SHAPE)
    pan_frames = make_frame_sequence(base, N_PAN_BEFORE + N_PAN_AFTER, SHIFT_PX)
    cut_target = make_multiplane_target(SHAPE, soft=True)

    sequence = list(pan_frames[:N_PAN_BEFORE])
    sequence.append(cut_target)
    sequence.extend(pan_frames[N_PAN_BEFORE:N_PAN_BEFORE + N_PAN_AFTER])
    return sequence


def main():
    check_patch_applied()
    frames = build_sequence()
    n_frames = len(frames)

    print(f"Sequence: frames 0-{N_PAN_BEFORE - 1} = realistic pan, frame {CUT_FRAME_INDEX} = scene cut "
          f"(unrelated synthetic content), frames {CUT_FRAME_INDEX + 1}-{n_frames - 1} = realistic pan resumes\n")

    cold = run_sequence(frames, warm_start=False)
    warm = run_sequence(frames, warm_start=True)

    print(f"{'Frame':<8}{'Content':<12}{'Cold final':<14}{'Cold cross-iter':<18}{'Warm cross-iter':<18}{'Speedup':<10}")
    worst_warm_iters = 0
    worst_warm_frame = None
    for f in range(n_frames):
        content = "CUT" if f == CUT_FRAME_INDEX else "pan"
        cold_final = cold[f]["history"][-1]
        threshold = cold_final - QUALITY_THRESHOLD_MARGIN_DB
        cold_cross = first_crossing(cold[f]["history"], threshold)
        warm_cross = first_crossing(warm[f]["history"], threshold)

        if warm_cross is None:
            speedup_str = "NEVER CONVERGED"
            effective_warm_iters = N_ITERS_BUDGET  # did not converge within budget -- worst case is the whole budget
        else:
            effective_warm_iters = warm_cross
            if cold_cross:
                speedup = cold_cross / warm_cross
                speedup_str = f"{speedup:.2f}x" if speedup >= 1.0 else f"{speedup:.2f}x (SLOWER than cold)"
            else:
                speedup_str = "n/a"

        if f > 0 and effective_warm_iters > worst_warm_iters:
            worst_warm_iters = effective_warm_iters
            worst_warm_frame = f

        print(f"{f:<8}{content:<12}{cold_final:<14.2f}{str(cold_cross):<18}{str(warm_cross):<18}{speedup_str:<10}")

    print(f"\nWorst-case warm-start iterations-to-threshold (frames 1+): {worst_warm_iters} "
          f"(frame {worst_warm_frame}, {'CUT' if worst_warm_frame == CUT_FRAME_INDEX else 'pan'})")

    cold_worst_iters = 0
    for f in range(n_frames):
        cold_final = cold[f]["history"][-1]
        threshold = cold_final - QUALITY_THRESHOLD_MARGIN_DB
        c = first_crossing(cold[f]["history"], threshold)
        if c and c > cold_worst_iters:
            cold_worst_iters = c
    print(f"Worst-case cold-start iterations-to-threshold (all frames, for reference): {cold_worst_iters}")

    if worst_warm_frame == CUT_FRAME_INDEX:
        print("\nThe scene cut is the worst case, as expected: warm-starting from an unrelated")
        print("phase costs the most iterations of any frame in this sequence. This is the number")
        print("that should size any iteration/frame-rate budget that assumes warm-starting --")
        print("not the steady-state pan speedup, which only applies between similar frames.")
        cut_warm_cross = first_crossing(warm[CUT_FRAME_INDEX]["history"],
                                         cold[CUT_FRAME_INDEX]["history"][-1] - QUALITY_THRESHOLD_MARGIN_DB)
        cut_cold_cross = first_crossing(cold[CUT_FRAME_INDEX]["history"],
                                         cold[CUT_FRAME_INDEX]["history"][-1] - QUALITY_THRESHOLD_MARGIN_DB)
        if cut_warm_cross and cut_cold_cross and cut_warm_cross > cut_cold_cross:
            print(f"Negative transfer confirmed: warm-start needed MORE iterations ({cut_warm_cross}) than")
            print(f"cold-start ({cut_cold_cross}) at the cut frame -- the mismatched starting phase actively")
            print("hurt convergence rather than just failing to help.")
        elif cut_warm_cross and cut_cold_cross:
            print(f"No negative transfer at the cut: warm ({cut_warm_cross} iters) still <= cold ({cut_cold_cross} iters),")
            print("just far less of an advantage than steady-state panning gives.")
    else:
        print(f"\nUnexpected: the worst case was a pan frame ({worst_warm_frame}), not the scene cut. Worth")
        print("double-checking the recovery frame right after the cut specifically.")

    recovery_frame = CUT_FRAME_INDEX + 1
    if recovery_frame < n_frames:
        rc_final = cold[recovery_frame]["history"][-1]
        rc_thresh = rc_final - QUALITY_THRESHOLD_MARGIN_DB
        rc_cold = first_crossing(cold[recovery_frame]["history"], rc_thresh)
        rc_warm = first_crossing(warm[recovery_frame]["history"], rc_thresh)
        print(f"\nRecovery frame ({recovery_frame}, first pan frame after the cut): cold={rc_cold} iters, "
              f"warm={rc_warm} iters -- checks whether warm-start bounces back immediately or is still")
        print("dragged down by the cut frame's unrelated converged phase.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, N_ITERS_BUDGET + 1), cold[CUT_FRAME_INDEX]["history"], label="Cold-start (cut frame)", marker="o")
    ax.plot(range(1, N_ITERS_BUDGET + 1), warm[CUT_FRAME_INDEX]["history"], label="Warm-start (cut frame)", marker="s")
    ax.axhline(cold[CUT_FRAME_INDEX]["history"][-1] - QUALITY_THRESHOLD_MARGIN_DB, color="gray", linestyle="--",
               label="Quality threshold")
    ax.set_xlabel("GS iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"Scene-cut frame ({CUT_FRAME_INDEX}): cold vs. warm-start convergence")
    ax.legend()
    plt.tight_layout()
    plt.savefig("temporal_warmstart_scenecut.png", dpi=130)
    print("\nSaved plot: temporal_warmstart_scenecut.png")


if __name__ == "__main__":
    main()
