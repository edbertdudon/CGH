"""
Stage 1 of the 10.5 sequence: is SGD's plateau at step 500 a real wall, or
just a fixed-learning-rate artifact? Run this before touching sequential-
vs-simultaneous or full resolution -- it decides which SGD config those
later experiments should even use.

Run on your 3060:
    python3 experiment_lr_schedule.py
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_multiplane_target
from retrieval_torch import multiplane_sgd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)
N_PLANES = 3
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_STEPS = 500


def main():
    print(f"Device: {DEVICE}")
    targets = make_multiplane_target(SHAPE, n_planes=N_PLANES, soft=True, sigma=3.0)

    runs = {
        "fixed lr=0.02 (baseline, last run)": dict(lr=0.02, lr_schedule=None),
        "fixed lr=0.05 (just try higher)": dict(lr=0.05, lr_schedule=None),
        "cosine decay from lr=0.05": dict(lr=0.05, lr_schedule="cosine"),
        "one-cycle (warmup then decay)": dict(lr=0.02, lr_schedule="onecycle"),
    }

    results = {}
    for name, kwargs in runs.items():
        _, history, _, lr_hist = multiplane_sgd(
            targets, DEPTHS_M, WAVELENGTH, DX, N_STEPS, device=DEVICE, seed=0, **kwargs
        )
        results[name] = history
        print(f"{name:35s} final PSNR: {history[-1]:.2f} dB  (last-50-step gain: "
              f"{history[-1]-history[-50]:+.2f} dB)")

    print()
    best = max(results, key=lambda k: results[k][-1])
    print(f"Best: {best} at {results[best][-1]:.2f} dB")
    print("Check the 'last-50-step gain' column: near-zero means that run has genuinely")
    print("plateaued; still-positive means it likely had more room and 500 steps wasn't enough.")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, history in results.items():
        ax.plot(range(1, N_STEPS + 1), history, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"SGD: does a learning-rate schedule break the plateau? ({SHAPE[0]}x{SHAPE[1]}, {N_PLANES} planes)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("lr_schedule_comparison.png", dpi=130)
    print("Saved plot: lr_schedule_comparison.png")


if __name__ == "__main__":
    main()
