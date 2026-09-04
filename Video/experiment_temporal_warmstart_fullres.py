"""
Follow-up (3) to the temporal warm-start experiment family: the full-
resolution check.

Everything so far (experiment_temporal_warmstart.py and its
_shift_sweep / _zeroiter / _padcrop / _scenecut follow-ups) ran at
512x512. Section 10.3's own history in this document is the direct
cautionary precedent: a large, clean sequential-rendering advantage at
512x512 shrank to ~0 dB at the actual 4,700x2,700 target. This reruns
the core warm-start-speedup question at the document's real target
resolution, with real content (near/mid/far, from targets.py, same as
10.6), using pad-crop (non-wrapping) motion rather than torch.roll --
avoiding the wraparound confound flagged in _padcrop.py, even though
that script found wraparound wasn't the full explanation for the
non-monotonic speedup curve.

Two shift magnitudes only (not the full 6-point sweep) to keep runtime
reasonable at full resolution: a small pan (~0.6% of frame width, same
relative magnitude as the gentlest 512x512 case) and a larger pan
(~12%, same relative magnitude as the 512x512 case that showed the
smallest -- but still real -- speedup). N_FRAMES and N_ITERS_BUDGET are
both reduced from the 512x512 scripts to keep total runtime to single-
digit minutes on a 3060; this is a scale check, not a full replication
of the shift sweep.

Uses the corrected multiplane_gs_torch history (off-by-one fix applied
earlier this session) -- these numbers are the first temporal-
warm-start results measured against the corrected history at any scale.

Run on your 3060 (expect several minutes):
    python3 experiment_temporal_warmstart_fullres.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from experiment_temporal_warmstart import check_patch_applied, run_sequence, first_crossing
from experiment_temporal_warmstart_padcrop import make_frame_sequence_padcrop

SHAPE = (2700, 4700)               # doc target resolution (N_y, N_x), same as 10.1c / 10.6
N_FRAMES = 5                        # reduced from 8 (512x512 scripts) to keep full-res runtime reasonable
N_ITERS_BUDGET = 40                 # 10.1c found full-res GS plateaus ~36-58 iters from cold; budget must cover that
QUALITY_THRESHOLD_MARGIN_DB = 0.5   # same definition as every other script in this family
SHIFT_FRACTIONS = [0.006, 0.117]    # match the smallest/largest fractions tested at 512x512 (3px and 60px of 512)


def average_speedup(frames):
    cold = run_sequence(frames, warm_start=False)
    warm = run_sequence(frames, warm_start=True)
    rows = []
    for f in range(len(frames)):
        cold_final = cold[f]["history"][-1]
        threshold = cold_final - QUALITY_THRESHOLD_MARGIN_DB
        cold_cross = first_crossing(cold[f]["history"], threshold)
        warm_cross = first_crossing(warm[f]["history"], threshold)
        rows.append((f, cold_final, cold_cross, warm_cross))
    speedups = [c / w for _, _, c, w in rows[1:] if c and w]
    avg = sum(speedups) / len(speedups) if speedups else None
    return rows, avg


def main():
    check_patch_applied()
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    t_start = time.time()
    base = make_realistic_multiplane_target(SHAPE)

    all_avgs = {}
    for frac in SHIFT_FRACTIONS:
        shift_px = round(frac * SHAPE[1])
        frames = make_frame_sequence_padcrop(base, N_FRAMES, shift_px)
        print(f"\n=== shift={shift_px}px ({frac*100:.1f}% of frame width), full resolution {SHAPE[1]}x{SHAPE[0]}, "
              f"real content, pad-crop (non-wrapping) motion ===")
        rows, avg = average_speedup(frames)
        print(f"{'Frame':<8}{'Cold final':<14}{'Cold cross-iter':<18}{'Warm cross-iter':<18}{'Speedup':<10}")
        for f, cold_final, cold_cross, warm_cross in rows:
            if f == 0:
                speedup_str = "1.00x"
            elif cold_cross and warm_cross:
                speedup_str = f"{cold_cross/warm_cross:.2f}x"
            else:
                speedup_str = "n/a"
            print(f"{f:<8}{cold_final:<14.2f}{str(cold_cross):<18}{str(warm_cross):<18}{speedup_str:<10}")
        avg_str = f"{avg:.2f}x" if avg is not None else "n/a"
        print(f"Average speedup (frames 1+): {avg_str}")
        all_avgs[shift_px] = avg

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} min")

    print("\n" + "=" * 70)
    print("FULL-RESOLUTION RESULT vs. 512x512 (for direct comparison):")
    ref_512 = {"0.6% (3px @512)": 14.86, "11.7% (60px @512)": 8.31}  # roll-based, corrected-history 512x512 shift sweep, from this session
    for (shift_px, avg), (label, ref) in zip(all_avgs.items(), ref_512.items()):
        avg_str = f"{avg:.2f}x" if avg is not None else "n/a (never converged within budget)"
        print(f"  {label}: 512x512 gave {ref}x; full-res ({shift_px}px) gives {avg_str}")
    print("=" * 70)

    survives = all(v is not None and v > 1.3 for v in all_avgs.values())
    if survives:
        print("\nWarm-start speedup survives the jump to full resolution -- unlike 10.3's sequential-")
        print("rendering advantage, which vanished at this same resolution. Real evidence, not yet")
        print("a green light for Gap 2: still only 2 shift points, reduced iteration/frame budget,")
        print("and the scene-cut worst case (see _scenecut.py) has not been retested at this scale.")
    else:
        print("\nSpeedup does NOT survive cleanly at full resolution at every shift tested -- matches")
        print("the 10.3 cautionary pattern (real at 512x512, gone/reduced at target resolution). Do")
        print("NOT treat the 512x512 shift-sweep numbers as representative of the real target without")
        print("this caveat attached.")


if __name__ == "__main__":
    main()
