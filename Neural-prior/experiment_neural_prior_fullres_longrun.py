"""
Follow-up to experiment_neural_prior_fullres_calibration.py: all four
learning rates tested there "settled" by a short-window swing metric
within 150 steps, but every one plateaued around 6.3-6.4 dB -- well
below both the original 512x512 result (~11 dB) and SGD's full-res
baseline (~13.8 dB). One run (lr=0.01) peaked at 10.26 dB then decayed
to 6.33 dB by step 150 -- a rise-then-decay shape, not simple noise
around a fixed point.

This runs a single seed for longer (500 steps, using the best-looking
calibration LR, 0.003) to see the full shape of the curve: does it
recover past the ~6.4 dB ceiling given more time, does it decay further,
or does it genuinely plateau there? Answering this before committing to
a 3-seed, ~3-hour full comparison at whatever budget turns out to be
right.

Run on your 3060 (expect ~35-40 min):
    python3 experiment_neural_prior_fullres_longrun.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_neural_prior

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_STEPS = 500
SEED = 0
LR = 3e-3


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_neural_prior(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                                 pad_factor=PAD_FACTOR, seed=SEED, lr=LR)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print(f"\nRunning {N_STEPS} steps at lr={LR}, seed={SEED}...")
    t0 = time.time()
    _, history, _, n_params = multiplane_neural_prior(
        targets, DEPTHS_M, WAVELENGTH, DX, N_STEPS, lr=LR, device=DEVICE,
        pad_factor=PAD_FACTOR, seed=SEED
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    print(f"\nDone: {elapsed:.1f}s ({elapsed/N_STEPS:.2f}s/step), {n_params:,} network params")
    print(f"Final: {history[-1]:.2f} dB")
    print(f"Peak: {max(history):.2f} dB at step {history.index(max(history))+1}")
    print(f"Min: {min(history):.2f} dB at step {history.index(min(history))+1}")

    checkpoints = [25, 50, 100, 150, 200, 300, 400, 500]
    print("\nProgress at checkpoints:")
    for c in checkpoints:
        if c <= len(history):
            print(f"   step {c}: {history[c-1]:.2f} dB")

    peak_idx = history.index(max(history))
    if peak_idx < len(history) - 50 and history[-1] < max(history) - 1.0:
        print("\nCurve peaks early and decays -- more steps make it WORSE, not better. The")
        print("network is not simply slow to converge; something in the optimization dynamics")
        print("actively degrades quality after an early peak. Early stopping at the peak, not a")
        print("longer budget, would be the way to get this network's best full-resolution number.")
    elif history[-1] > max(history[:max(1,len(history)-50)]) - 0.2:
        print("\nCurve is still improving or holding its peak at the end of the run -- a longer")
        print("budget may continue to help; this is now a candidate for the full 3-seed comparison")
        print("at this step count or slightly more.")
    else:
        print("\nCurve is roughly flat/noisy without a clear trend -- report as plateaued at this")
        print("quality level, whatever it is relative to the peak.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, N_STEPS + 1), history, linewidth=1)
    ax.axhline(max(history), color="gray", linestyle="--", linewidth=0.8, label=f"peak {max(history):.2f} dB")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"Neural-prior full-resolution long run, lr={LR}, seed={SEED}, {N_STEPS} steps")
    ax.legend()
    plt.tight_layout()
    plt.savefig("neural_prior_fullres_longrun.png", dpi=130)
    print("Saved plot: neural_prior_fullres_longrun.png")


if __name__ == "__main__":
    main()
