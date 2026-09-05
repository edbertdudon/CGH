"""
Audit follow-up (10.9/A.13): the learning-rate sweep in 10.2 ("four
materially different optimization strategies, under 0.6dB of spread")
is the single most exposed finding in this document -- single-seed AND
never resolution-checked, the same double-gap that broke 10.3 and 10.8.
This fixes both at once: full target resolution (2,700x4,700, up from
512x512) and multiple seeds per strategy (up from one), rather than
fixing them one at a time.

Same four strategies as the original test, using multiplane_sgd's
existing lr_schedule support:
  - fixed lr=0.02
  - fixed lr=0.05
  - cosine decay from lr=0.05
  - one-cycle (warmup then decay) from lr=0.02

N_STEPS raised generously (800, up from 500) given this project's
established pattern of full-resolution convergence needing more
iterations than small-scale runs suggested; "still improving in the
final 100 steps" is checked explicitly per condition rather than
assumed settled.

This is the finding cited in Gap 7/10.2 as ruling out optimizer tuning
as the limiting factor on the ~11-14dB quality ceiling -- if the spread
across strategies turns out to be larger or seed-dependent at the real
resolution, that specific piece of evidence needs to be revised.

Run on your 3060 (expect a while -- full-resolution SGD, 12 runs):
    python3 experiment_lr_sweep_seed_check.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_sgd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_STEPS = 800
N_SEEDS = 3

STRATEGIES = {
    "fixed lr=0.02": dict(lr=0.02, lr_schedule=None),
    "fixed lr=0.05": dict(lr=0.05, lr_schedule=None),
    "cosine decay from lr=0.05": dict(lr=0.05, lr_schedule="cosine"),
    "one-cycle from lr=0.02": dict(lr=0.02, lr_schedule="onecycle"),
}


def still_improving(history, window=100, tol=0.1):
    if len(history) < window:
        return False
    return (history[-1] - history[-window]) > tol


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_sgd(targets, DEPTHS_M, WAVELENGTH, DX, 2, device=DEVICE, seed=0)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    results = {name: [] for name in STRATEGIES}
    for name, kwargs in STRATEGIES.items():
        print(f"\n{name}:")
        for seed in range(N_SEEDS):
            t0 = time.time()
            _, history, _, _ = multiplane_sgd(
                targets, DEPTHS_M, WAVELENGTH, DX, N_STEPS, device=DEVICE, seed=seed, **kwargs
            )
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            final = history[-1]
            improving = still_improving(history)
            results[name].append({"seed": seed, "final": final, "history": history, "improving": improving})
            print(f"   seed={seed}: final={final:.2f} dB, {elapsed:.1f}s"
                  f"{' STILL IMPROVING -- N_STEPS may be too low' if improving else ''}")

    print("\n" + "=" * 90)
    print(f"{'Strategy':<28}{'Seeds (dB)':<32}{'Avg':<10}{'Spread'}")
    all_finals = []
    for name, runs in results.items():
        finals = [r["final"] for r in runs]
        all_finals.extend(finals)
        avg = sum(finals) / len(finals)
        spread = max(finals) - min(finals)
        print(f"{name:<28}{str([round(f,2) for f in finals]):<32}{avg:<10.2f}{spread:.2f}")
    print("=" * 90)

    overall_spread = max(all_finals) - min(all_finals)
    any_improving = any(r["improving"] for runs in results.values() for r in runs)

    print(f"\nOriginal single-seed, 512x512 finding: under 0.6dB spread across 4 strategies.")
    print(f"This test (N_SEEDS={N_SEEDS}, full resolution): overall spread across all "
          f"{len(all_finals)} runs = {overall_spread:.2f} dB")
    if any_improving:
        print("WARNING: at least one run was still improving at N_STEPS -- consider a longer budget")
        print("before fully trusting the spread number above.")

    if overall_spread < 1.0:
        print("\nSpread remains small at full resolution across seeds -- the original 'optimizer tuning")
        print("doesn't matter' conclusion holds up under the corrected methodology.")
    elif overall_spread < 2.5:
        print("\nSpread is larger than originally reported but still modest -- the qualitative conclusion")
        print("(no strategy is a clear winner) likely still holds, but the original <0.6dB precision claim")
        print("does not; report the wider range instead.")
    else:
        print("\nSpread is substantially larger than originally reported -- the 'optimizer tuning doesn't")
        print("matter' conclusion does NOT hold up as stated; seed and/or resolution meaningfully change")
        print("which strategy performs best, and this piece of Gap 7's evidence needs revision.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, runs in results.items():
        for r in runs:
            ax.plot(range(1, N_STEPS + 1), r["history"], alpha=0.5,
                     label=name if r["seed"] == 0 else None)
    ax.set_xlabel("SGD step")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"LR-sweep seed/full-res check: {N_SEEDS} seeds x 4 strategies, {SHAPE[1]}x{SHAPE[0]}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("lr_sweep_seed_check.png", dpi=130)
    print("Saved plot: lr_sweep_seed_check.png")


if __name__ == "__main__":
    main()
