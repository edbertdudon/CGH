"""
Full-resolution calibration pass for the neural-prior comparison (10.2),
flagged in the v20 audit as "attempted, found untestable at current
cost" after a 20-step probe at default settings (lr=1e-3) showed
unstable, non-converging quality (PSNR oscillating 4.19-6.36 dB) and
~4.6s/step (~10x SGD's per-step cost).

Before declaring the comparison genuinely impossible, check the obvious
confound first: the network's default learning rate was tuned (if at
all) for the 512x512 small-scale test, not for a 2700x4700 target -- the
same "settings tuned for a smaller resolution" trap that has bitten
other experiments in this project (temporal warm-start, LR sweep, etc).
This sweeps learning rate at full resolution, single seed, over a short
but diagnostic budget (150 steps -- long enough to distinguish "still
oscillating" from "was just slow to settle") before committing to a
full 3-seed run at whichever setting looks stable.

Run on your 3060 (expect ~45-60 min for 4 LRs x 150 steps at full res):
    python3 experiment_neural_prior_fullres_calibration.py
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
N_STEPS = 150
SEED = 0
LRS = [3e-4, 1e-3, 3e-3, 1e-2]


def oscillation_score(history, window=20):
    # average absolute step-to-step swing over the final window, as a
    # fraction of the total range covered -- large relative swing late
    # in the run means "still bouncing", not "still climbing"
    tail = history[-window:]
    swings = [abs(tail[i] - tail[i - 1]) for i in range(1, len(tail))]
    avg_swing = sum(swings) / len(swings)
    rng = max(history) - min(history) if max(history) != min(history) else 1e-9
    return avg_swing, avg_swing / rng


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_neural_prior(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                                 pad_factor=PAD_FACTOR, seed=SEED)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    results = {}
    for lr in LRS:
        t0 = time.time()
        _, history, _, n_params = multiplane_neural_prior(
            targets, DEPTHS_M, WAVELENGTH, DX, N_STEPS, lr=lr, device=DEVICE,
            pad_factor=PAD_FACTOR, seed=SEED
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        avg_swing, rel_swing = oscillation_score(history)
        results[lr] = {"history": history, "final": history[-1], "elapsed": elapsed,
                        "avg_swing": avg_swing, "rel_swing": rel_swing,
                        "min": min(history), "max": max(history)}
        print(f"\nlr={lr}: final={history[-1]:.2f} dB, range=[{min(history):.2f}, {max(history):.2f}] dB, "
              f"{elapsed:.1f}s ({elapsed/N_STEPS:.2f}s/step)")
        print(f"   last-20-step avg swing: {avg_swing:.3f} dB ({rel_swing*100:.0f}% of total range)"
              f"{' -- STILL OSCILLATING' if rel_swing > 0.15 else ' -- settled'}")

    print("\n" + "=" * 90)
    print(f"{'LR':<10}{'Final dB':<12}{'Range':<20}{'s/step':<10}{'Tail swing (rel)':<20}{'Verdict'}")
    for lr, r in results.items():
        verdict = "OSCILLATING" if r["rel_swing"] > 0.15 else "settled"
        range_str = f"[{r['min']:.1f}, {r['max']:.1f}]"
        print(f"{lr:<10}{r['final']:<12.2f}{range_str:<20}"
              f"{r['elapsed']/N_STEPS:<10.2f}{r['rel_swing']*100:<19.0f}{verdict}")
    print("=" * 90)

    settled = {lr: r for lr, r in results.items() if r["rel_swing"] <= 0.15}
    if settled:
        best_lr = max(settled, key=lambda lr: settled[lr]["final"])
        print(f"\nAt least one LR settles within {N_STEPS} steps. Best settled LR: {best_lr} "
              f"({settled[best_lr]['final']:.2f} dB). Use this for the full 3-seed comparison, "
              f"with a generous step budget given full-resolution GS/SGD both needed more steps "
              f"than their small-scale runs suggested.")
    else:
        print(f"\nNo LR tested settles within {N_STEPS} steps -- every one is still meaningfully")
        print("bouncing in its final 20 steps. This suggests the instability is not simply a learning")
        print("rate problem; a longer budget or architecture change may be needed, or the network's")
        print("training dynamics may genuinely not suit this resolution/loss combination.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    for lr, r in results.items():
        ax.plot(range(1, N_STEPS + 1), r["history"], label=f"lr={lr}", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"Neural-prior LR calibration, full resolution ({SHAPE[1]}x{SHAPE[0]}), seed={SEED}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("neural_prior_fullres_calibration.png", dpi=130)
    print("Saved plot: neural_prior_fullres_calibration.png")


if __name__ == "__main__":
    main()
