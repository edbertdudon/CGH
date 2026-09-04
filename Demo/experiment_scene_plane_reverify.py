"""
10.8 item: the photo-like scene plane showed a +3.0 dB sequential
advantage in 10.6 that was never independently re-verified -- unlike
every other full-resolution finding in this document, it's a single-run
observation. This reruns just that comparison (simultaneous vs.
sequential, scene plane only) across multiple random seeds, to check
whether the effect is consistent (real) or seed-dependent (noise).

Run on your 3060:
    python3 experiment_scene_plane_reverify.py
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch
from metrics import psnr_intensity as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 16
SEEDS = [0, 1, 2, 3, 4]

SCENE_IDX = 2   # far: photo-like scene, the plane being re-verified
SCENE_Z = DEPTHS_M[SCENE_IDX]


def main():
    print(f"Device: {DEVICE}")
    targets = make_realistic_multiplane_target(SHAPE)
    scene_target = targets[SCENE_IDX]

    deltas = []
    print(f"{'seed':<6}{'simultaneous':<16}{'sequential':<14}{'delta':<10}")
    for seed in SEEDS:
        # simultaneous: solve all 3 planes together, same seed, read off scene plane's quality
        _, _, cps_sim = multiplane_gs_torch(
            targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, seed=seed, pad_factor=PAD_FACTOR
        )
        final_sim = max(cps_sim.keys())
        q_sim = psnr_np(cps_sim[final_sim][SCENE_IDX], scene_target)

        # sequential: scene plane solved alone, same seed
        _, _, cps_seq = multiplane_gs_torch(
            [scene_target], [SCENE_Z], WAVELENGTH, DX, N_ITERS, device=DEVICE, seed=seed, pad_factor=PAD_FACTOR
        )
        final_seq = max(cps_seq.keys())
        q_seq = psnr_np(cps_seq[final_seq][0], scene_target)

        delta = q_seq - q_sim
        deltas.append(delta)
        print(f"{seed:<6}{q_sim:<16.2f}{q_seq:<14.2f}{delta:+.2f}")

    deltas = np.array(deltas)
    print()
    print("=" * 50)
    print(f"Mean delta across {len(SEEDS)} seeds: {deltas.mean():+.2f} dB")
    print(f"Std dev across seeds: {deltas.std():.2f} dB")
    print(f"Range: {deltas.min():+.2f} to {deltas.max():+.2f} dB")
    print()
    print("Original single-run observation (10.6): +3.0 dB")
    if deltas.std() < 1.0 and deltas.mean() > 1.0:
        print("Consistent across seeds, well above noise level -- the scene plane's")
        print("sequential advantage appears to be REAL, a genuine content-dependent")
        print("exception to the otherwise-consistent no-advantage finding.")
    elif abs(deltas.mean()) < 1.0 or deltas.std() > abs(deltas.mean()):
        print("Delta is small relative to seed-to-seed variation -- the original +3.0 dB")
        print("was likely a lucky/unlucky seed, not a real effect. Consistent with the")
        print("no-advantage finding after all.")
    else:
        print("Mixed signal -- real but noisier than other findings in this document.")
    print("=" * 50)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(SEEDS)), deltas, tick_label=[str(s) for s in SEEDS])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(deltas.mean(), color="red", linestyle="--", label=f"mean = {deltas.mean():+.2f} dB")
    ax.axhline(3.0, color="gray", linestyle=":", label="original single-run value (+3.0 dB)")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Sequential - Simultaneous (dB)")
    ax.set_title("Scene plane sequential advantage, across seeds")
    ax.legend()
    plt.tight_layout()
    plt.savefig("scene_plane_reverify.png", dpi=130)
    print("Saved plot: scene_plane_reverify.png")


if __name__ == "__main__":
    main()
