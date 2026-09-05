"""
Every warm-start and scene-cut result so far (10.5, A.10) was built from
ONE base scene from make_realistic_multiplane_target, varied only by
constant pixel shift or plane permutation. That leaves a real gap open:
is the finding actually about "similar vs. different content," or an
artifact of reusing the same underlying pixels every time? This tests
against genuinely independent content instead.

METHODOLOGY CAVEAT -- read before trusting the results:
This is still NOT real motion-capture video or recorded UI-interaction
footage -- neither is available. What it IS: procedurally-generated
scenes that are independent of each other (not derived from one shared
source via shift/permutation), and a smoother, momentum-based random-walk
motion model with per-plane parallax (near moves faster than far, a real
optical effect of motion at different depths) instead of the earlier
constant-velocity shift. This is a real step up in realism, not a claim
of validation against actual product content.

Two things are retested, both using the plateau-verified true-convergence
methodology established in 10.7/A.9/A.10 (no fixed small budget):

  A) Scene-cut robustness across independently-generated scene pairs
     (not shifts/permutations of each other)
  B) Steady-state warm-starting across a parallax-motion sequence
     (smooth random-walk velocity per plane, not constant shift)

Small scale first (512x512), matching this project's usual pattern.

Run on your 3060:
    python3 experiment_independent_content.py
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_BUDGET = 100
N_RANDOM_SEEDS = 3
PLATEAU_HOLD = 5
PLATEAU_EPS_DB = 0.1
N_INDEPENDENT_SCENES = 4
N_MOTION_FRAMES = 8
WORDS = ["ALERT", "BATTERY", "MESSAGE", "TURN LEFT", "12:04", "LOW SIGNAL", "OK", "RECORDING"]


def detect_plateau(history, hold=PLATEAU_HOLD, eps=PLATEAU_EPS_DB):
    for i in range(hold, len(history) + 1):
        window = history[i - hold:i]
        if max(window) - min(window) < eps and abs(window[-1] - sum(window) / hold) < eps:
            return i
    return None


def render_plane(kind, shape, rng):
    """
    Procedurally generates ONE genuinely independent plane's content.
    kind: 'icon' (near), 'text' (mid), or 'photo' (far) -- matches this
    project's established near/mid/far content roles (10.6), but every
    call with a different rng produces different content, not a
    shift/permutation of a shared source.
    """
    h, w = shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("black")

    if kind == "icon":
        cx, cy = rng.uniform(0.3, 0.7, size=2)
        size = rng.uniform(0.08, 0.18)
        shape_choice = rng.choice(["circle", "square", "triangle"])
        if shape_choice == "circle":
            ax.add_patch(Circle((cx, cy), size, color="white"))
        elif shape_choice == "square":
            ax.add_patch(Rectangle((cx - size, cy - size), 2 * size, 2 * size, color="white"))
        else:
            ax.add_patch(RegularPolygon((cx, cy), 3, radius=size, color="white"))
    elif kind == "text":
        word = rng.choice(WORDS)
        fontsize = rng.uniform(28, 44)
        x, y = rng.uniform(0.15, 0.55), rng.uniform(0.4, 0.6)
        ax.text(x, y, word, color="white", fontsize=fontsize, fontfamily="sans-serif")
    elif kind == "photo":
        # smooth random field: sum of a few random 2D Gaussian blobs, independent per call
        yy, xx = np.mgrid[0:1:complex(h), 0:1:complex(w)]
        field = np.zeros((h, w))
        n_blobs = rng.integers(4, 9)
        for _ in range(n_blobs):
            bx, by = rng.uniform(0, 1, size=2)
            sx, sy = rng.uniform(0.05, 0.25, size=2)
            amp = rng.uniform(0.3, 1.0)
            field += amp * np.exp(-(((xx - bx) ** 2) / (2 * sx ** 2) + ((yy - by) ** 2) / (2 * sy ** 2)))
        field = field / field.max()
        ax.imshow(field, cmap="gray", extent=[0, 1, 0, 1], vmin=0, vmax=1)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, 0].astype(np.float64) / 255.0
    plt.close(fig)
    return buf[:h, :w]


def make_independent_scene(shape, seed):
    rng = np.random.default_rng(seed)
    return [render_plane("icon", shape, rng), render_plane("text", shape, rng), render_plane("photo", shape, rng)]


def pad_crop_shift_subpixel(plane, dx_px, dy_px):
    """Non-wrapping shift via simple integer rounding (sub-pixel motion
    rounded to nearest pixel -- adequate for this test's purposes, not a
    claim of sub-pixel-accurate rendering)."""
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


def make_parallax_motion_sequence(base_scene, n_frames, seed):
    """
    Smooth random-walk (momentum-based) velocity, independent per plane,
    scaled by a parallax factor so nearer planes move faster than farther
    ones -- a real optical effect of translational motion at different
    depths, not modeled in the earlier constant-shift test.
    """
    rng = np.random.default_rng(seed)
    parallax_factors = [1.0, 0.6, 0.3]  # near, mid, far -- near moves most
    frames = [base_scene]
    velocities = [np.array([0.0, 0.0]) for _ in base_scene]
    current = base_scene
    for _ in range(n_frames - 1):
        next_frame = []
        for i, plane in enumerate(current):
            velocities[i] = velocities[i] * 0.7 + rng.normal(0, 3.0, size=2) * parallax_factors[i]
            shifted = pad_crop_shift_subpixel(plane, velocities[i][0], velocities[i][1])
            next_frame.append(shifted)
        frames.append(next_frame)
        current = next_frame
    return frames


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
    plateau = detect_plateau(history)
    return {"phase": phase, "history": history, "plateau": plateau, "final": history[-1]}


def part_a_independent_scene_cuts():
    print("=" * 90)
    print("PART A: Scene-cut robustness across genuinely independent content")
    print("=" * 90)
    scenes = [make_independent_scene(SHAPE, seed=100 + i) for i in range(N_INDEPENDENT_SCENES)]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    results = []
    for a, b in pairs:
        print(f"\nCut: scene {a} -> scene {b}")
        donor = solve(scenes[a], seed=0)
        cold_runs = [solve(scenes[b], seed=s) for s in range(N_RANDOM_SEEDS)]
        cold_plateaus = [r["plateau"] for r in cold_runs if r["plateau"] is not None]
        cold_avg = sum(cold_plateaus) / len(cold_plateaus) if cold_plateaus else None
        warm = solve(scenes[b], seed=0, init_phase=donor["phase"])
        warm_status = warm["plateau"] if warm["plateau"] is not None else f"NEGATIVE TRANSFER (no plateau in {N_ITERS_BUDGET})"
        cold_str = f"{min(cold_plateaus)}-{max(cold_plateaus)} (avg {cold_avg:.1f})" if cold_avg else "did not plateau"
        print(f"   cold: {cold_str} | warm: {warm_status}")
        results.append({"pair": f"{a}->{b}", "cold_avg": cold_avg, "warm": warm["plateau"]})

    negative = sum(1 for r in results if r["warm"] is None)
    print(f"\n{negative}/{len(results)} pairs showed negative transfer on genuinely independent content.")
    return results


def part_b_parallax_motion():
    print("\n" + "=" * 90)
    print("PART B: Steady-state warm-starting under parallax random-walk motion")
    print("=" * 90)
    base = make_independent_scene(SHAPE, seed=200)

    for seed in range(N_RANDOM_SEEDS):
        print(f"\nMotion sequence, seed {seed}...")
        frames = make_parallax_motion_sequence(base, N_MOTION_FRAMES, seed=300 + seed)
        cold_plateaus, warm_plateaus = [], []
        prev_phase = None
        for f, frame in enumerate(frames):
            cold = solve(frame, seed=f)
            warm = solve(frame, seed=f, init_phase=prev_phase) if prev_phase is not None else cold
            cold_plateaus.append(cold["plateau"])
            warm_plateaus.append(warm["plateau"] if prev_phase is not None else cold["plateau"])
            prev_phase = cold["phase"]  # chain from COLD's own converged phase, matching 10.5's frame-to-frame design
        print(f"   cold per-frame plateaus: {cold_plateaus}")
        print(f"   warm per-frame plateaus: {warm_plateaus}")

    print("\nCompare the per-frame warm plateaus (frames 1+) against cold's typical range above.")
    print("A consistent reduction under real parallax motion (not just uniform constant shift)")
    print("would confirm 10.5's finding generalizes beyond the original simplified motion model.")


def main():
    print(f"Device: {DEVICE}\n")
    part_a_independent_scene_cuts()
    part_b_parallax_motion()


if __name__ == "__main__":
    main()
