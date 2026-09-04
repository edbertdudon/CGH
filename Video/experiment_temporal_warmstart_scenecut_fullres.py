"""
Full-resolution retest of experiment_temporal_warmstart_scenecut.py's
negative-transfer finding: at 512x512, warm-starting from an unrelated
scene's converged phase never reached threshold quality within the
iteration budget, while cold-start solved the same (simpler synthetic)
content in 3 iterations -- the worst case in that whole experiment
family. That result sizes any iteration budget that assumes warm-
starting, so it needs to survive the same resolution jump the steady-
state pan speedup was just checked against (see _fullres.py).

Sequence: 2 pan frames (real content, pad-crop motion, small shift) ->
1 scene-cut frame (unrelated synthetic disc/ring/checker content) -> 2
more pan frames (recovery check). Shorter than the 512x512 version (7
frames) to keep full-resolution runtime reasonable -- this is a scale
check on the worst-case finding, not a full replication.

Uses the corrected multiplane_gs_torch history (off-by-one fix applied
earlier this session).

Run on your 3060 (expect a few minutes):
    python3 experiment_temporal_warmstart_scenecut_fullres.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, make_multiplane_target
import experiment_temporal_warmstart as ws
from experiment_temporal_warmstart import check_patch_applied, first_crossing
from experiment_temporal_warmstart_padcrop import make_frame_sequence_padcrop

SHAPE = (2700, 4700)
SHIFT_PX = 28              # same ~0.6% shift used in _fullres.py's small-shift case
N_ITERS_BUDGET = 40         # matches _fullres.py
QUALITY_THRESHOLD_MARGIN_DB = 0.5

# run_sequence() reads N_ITERS_BUDGET as a module-level global at call time --
# override it here so this script's budget actually takes effect (it silently
# didn't on the first run, which used the imported module's default of 30).
ws.N_ITERS_BUDGET = N_ITERS_BUDGET
run_sequence = ws.run_sequence
CUT_FRAME_INDEX = 2         # frames 0,1 = pan; 2 = cut; 3,4 = pan resumes


def build_sequence():
    base = make_realistic_multiplane_target(SHAPE)
    pan_frames = make_frame_sequence_padcrop(base, 4, SHIFT_PX)  # frames 0,1 before cut; 2,3 after (indices 2,3 map to sequence positions 3,4)
    cut_target = make_multiplane_target(SHAPE, soft=True)

    sequence = list(pan_frames[:2])
    sequence.append(cut_target)
    sequence.extend(pan_frames[2:4])
    return sequence


def main():
    check_patch_applied()
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    t_start = time.time()
    frames = build_sequence()
    n_frames = len(frames)

    print(f"Sequence: frames 0-1 = pan, frame {CUT_FRAME_INDEX} = scene cut, "
          f"frames {CUT_FRAME_INDEX+1}-{n_frames-1} = pan resumes. "
          f"Full resolution {SHAPE[1]}x{SHAPE[0]}, real content, pad-crop motion.\n")

    cold = run_sequence(frames, warm_start=False)
    warm = run_sequence(frames, warm_start=True)

    print(f"{'Frame':<8}{'Content':<12}{'Cold final':<14}{'Cold cross-iter':<18}{'Warm cross-iter':<18}{'Speedup':<10}")
    for f in range(n_frames):
        content = "CUT" if f == CUT_FRAME_INDEX else "pan"
        cold_final = cold[f]["history"][-1]
        threshold = cold_final - QUALITY_THRESHOLD_MARGIN_DB
        cold_cross = first_crossing(cold[f]["history"], threshold)
        warm_cross = first_crossing(warm[f]["history"], threshold)
        if warm_cross is None:
            speedup_str = "NEVER CONVERGED"
        elif cold_cross:
            speedup = cold_cross / warm_cross
            speedup_str = f"{speedup:.2f}x" if speedup >= 1.0 else f"{speedup:.2f}x (SLOWER than cold)"
        else:
            speedup_str = "n/a"
        print(f"{f:<8}{content:<12}{cold_final:<14.2f}{str(cold_cross):<18}{str(warm_cross):<18}{speedup_str:<10}")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} min")

    cut_final = cold[CUT_FRAME_INDEX]["history"][-1]
    cut_thresh = cut_final - QUALITY_THRESHOLD_MARGIN_DB
    cut_cold = first_crossing(cold[CUT_FRAME_INDEX]["history"], cut_thresh)
    cut_warm = first_crossing(warm[CUT_FRAME_INDEX]["history"], cut_thresh)

    print("\n" + "=" * 70)
    if cut_warm is None:
        print(f"Negative transfer CONFIRMED at full resolution: warm-start never reached")
        print(f"threshold within {N_ITERS_BUDGET} iterations at the cut frame; cold-start needed only")
        print(f"{cut_cold}. Matches the 512x512 finding -- this is not a small-scale artifact.")
    elif cut_cold and cut_warm > cut_cold:
        print(f"Negative transfer confirmed but bounded at full resolution: warm-start needed")
        print(f"{cut_warm} iterations vs. cold-start's {cut_cold} at the cut frame -- slower, not")
        print(f"a budget blowout like the 512x512 case, but still the wrong direction.")
    else:
        print(f"No negative transfer at full resolution: warm ({cut_warm}) <= cold ({cut_cold}) at the cut")
        print(f"frame. Does NOT replicate the 512x512 finding -- worth flagging as scale-dependent")
        print(f"rather than assuming the smaller-scale worst case still applies.")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, N_ITERS_BUDGET + 1), cold[CUT_FRAME_INDEX]["history"], label="Cold-start (cut frame)", marker="o")
    ax.plot(range(1, N_ITERS_BUDGET + 1), warm[CUT_FRAME_INDEX]["history"], label="Warm-start (cut frame)", marker="s")
    ax.axhline(cut_thresh, color="gray", linestyle="--", label="Quality threshold")
    ax.set_xlabel("GS iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"Full-resolution scene-cut frame: cold vs. warm-start convergence")
    ax.legend()
    plt.tight_layout()
    plt.savefig("temporal_warmstart_scenecut_fullres.png", dpi=130)
    print("Saved plot: temporal_warmstart_scenecut_fullres.png")


if __name__ == "__main__":
    main()
