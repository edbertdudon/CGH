"""
Follow-up to experiment_independent_content.py, fixing the detector
issue flagged before trusting its results.

That script's detect_plateau() only checks LOCAL flatness in a small
window (5 consecutive iterations within 0.1dB of their own trailing
average) starting from the earliest possible check point (iteration 5)
-- it never verifies that window matches where the curve actually ends
up. Warm-started runs hit that literal floor (exactly iteration 5)
almost every single time in the original run, which is the same shape
of false-positive that flipped the donor-seeding result from a 64% win
to a real loss once given a longer budget and a final-value-anchored
detector.

This reruns both parts with two fixes:
  1. A much longer budget (N_ITERS_BUDGET=200, up from 100) so any run
     that was still improving after a brief early flat patch has room
     to show it.
  2. The validated find_plateau() from Demo/convergence.py (anchored to
     the run's OWN actual final value, with an explicit still_rising
     check) instead of the local-only detect_plateau(), so a premature
     flat window can't be mistaken for genuine convergence.

Same scenes/motion model/seeds as the original script, small scale
(512x512) -- this is cheap enough at this resolution to just rerun with
a much larger safety margin rather than estimate one.

Run on your 3060:
    python3 experiment_independent_content_longrun.py
"""
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon

from fft_counter_torch import counter
from retrieval_torch import multiplane_gs_torch
from convergence import find_plateau_robust as find_plateau
from experiment_independent_content import (
    render_plane, make_independent_scene, pad_crop_shift_subpixel, make_parallax_motion_sequence,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_BUDGET = 200            # up from 100 -- see docstring
N_RANDOM_SEEDS = 3
N_INDEPENDENT_SCENES = 4
N_MOTION_FRAMES = 8


def solve(scene, seed, init_phase=None):
    counter.reset()
    t0 = time.time()
    phase, history, _ = multiplane_gs_torch(
        scene, DEPTHS_M, WAVELENGTH, DX, N_ITERS_BUDGET,
        device=DEVICE, seed=seed, pad_factor=PAD_FACTOR, init_phase=init_phase
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    plateau, final_ref, rising = find_plateau(history)
    return {"phase": phase, "history": history, "plateau": plateau, "final": history[-1], "rising": rising}


def part_a_independent_scene_cuts():
    print("=" * 90)
    print("PART A (long-run, final-value-anchored detector): scene-cut robustness, independent content")
    print("=" * 90)
    scenes = [make_independent_scene(SHAPE, seed=100 + i) for i in range(N_INDEPENDENT_SCENES)]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    results = []
    for a, b in pairs:
        print(f"\nCut: scene {a} -> scene {b}")
        donor = solve(scenes[a], seed=0)
        print(f"   donor: plateau={donor['plateau']}, final={donor['final']:.2f} dB"
              f"{' STILL RISING' if donor['rising'] else ''}")
        cold_runs = [solve(scenes[b], seed=s) for s in range(N_RANDOM_SEEDS)]
        cold_plateaus = [r["plateau"] for r in cold_runs]
        cold_avg = sum(cold_plateaus) / len(cold_plateaus)
        cold_rising = [s for s, r in enumerate(cold_runs) if r["rising"]]
        warm = solve(scenes[b], seed=0, init_phase=donor["phase"])
        cold_str = f"{min(cold_plateaus)}-{max(cold_plateaus)} (avg {cold_avg:.1f})"
        print(f"   cold: {cold_str}{' [seeds still rising: ' + str(cold_rising) + ']' if cold_rising else ''}")
        print(f"   warm: {warm['plateau']}, final={warm['final']:.2f} dB"
              f"{' STILL RISING' if warm['rising'] else ''}")
        results.append({"pair": f"{a}->{b}", "cold_avg": cold_avg, "warm": warm["plateau"],
                         "warm_rising": warm["rising"]})

    print("\n" + "-" * 90)
    for r in results:
        beat = "beats cold avg" if r["warm"] < r["cold_avg"] else "does NOT beat cold avg"
        flag = " [WARM STILL RISING -- unreliable]" if r["warm_rising"] else ""
        print(f"  {r['pair']}: warm={r['warm']} vs cold avg={r['cold_avg']:.1f} -- {beat}{flag}")
    return results


def part_b_parallax_motion():
    print("\n" + "=" * 90)
    print("PART B (long-run, final-value-anchored detector): parallax motion warm-starting")
    print("=" * 90)
    base = make_independent_scene(SHAPE, seed=200)

    for seed in range(N_RANDOM_SEEDS):
        print(f"\nMotion sequence, seed {seed}...")
        frames = make_parallax_motion_sequence(base, N_MOTION_FRAMES, seed=300 + seed)
        cold_plateaus, warm_plateaus, warm_rising_flags = [], [], []
        prev_phase = None
        for f, frame in enumerate(frames):
            cold = solve(frame, seed=f)
            warm = solve(frame, seed=f, init_phase=prev_phase) if prev_phase is not None else cold
            cold_plateaus.append(cold["plateau"])
            warm_plateaus.append(warm["plateau"] if prev_phase is not None else cold["plateau"])
            warm_rising_flags.append(warm["rising"] if prev_phase is not None else False)
            prev_phase = cold["phase"]
        print(f"   cold per-frame plateaus: {cold_plateaus}")
        print(f"   warm per-frame plateaus: {warm_plateaus}")
        if any(warm_rising_flags):
            still_rising_frames = [f for f, r in enumerate(warm_rising_flags) if r]
            print(f"   WARNING: warm-start still rising at frames {still_rising_frames} -- "
                  f"those plateau numbers are not reliable yet")


def main():
    print(f"Device: {DEVICE}\n")
    part_a_independent_scene_cuts()
    part_b_parallax_motion()


if __name__ == "__main__":
    main()
