"""
Full 3-seed neural-prior comparison at full resolution -- the last open
item from v20/v21's audit (Appendix A.14.3 marked this "untested, not
confirmed either way").

Two prior diagnostic runs changed what "fair comparison" means here.
A 4-LR calibration sweep (150 steps, 1 seed) and a longer 500-step
single run both showed the same shape: quality peaks early (step ~30),
then decays and locks onto a stable, exactly-flat plateau well below
the peak (6.34 dB from step ~100 through step 500, peak was 8.18 dB at
step 29). More steps make the full-resolution result WORSE, not better,
and then it gets stuck -- this is a real collapse, not noise, so
comparing SGD/GS's converged FINAL quality against neural-prior's final
quality would be comparing apples to a network that's actively forgotten
its own best answer.

Instead, this reports each seed's BEST-EVER quality reached during
training (with the step it occurred at), which is the fair comparison:
what the network could deliver with checkpoint selection, the way you'd
actually have to use it given this behavior. This is a different metric
than the final-value-at-convergence standard used everywhere else in
this document, and is labeled as such rather than presented as directly
equivalent.

N_STEPS=150 per seed, matching the calibration sweep's budget (long
enough to see the collapse and its stable floor, per the 500-step run).
lr=0.003, the best-performing rate from calibration.

Run on your 3060 (expect ~35-40 min, 3 seeds x 150 steps):
    python3 experiment_neural_prior_fullres_3seed.py
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
LR = 3e-3
SEEDS = [0, 1, 2]

# Established full-resolution, multi-seed baselines from this project's
# own prior work (10.1 GS baseline, LR-sweep SGD baseline) -- not
# rerun here, just quoted for comparison.
GS_BASELINE_DB = (11.47, 12.00)   # range across 5 seeds, 10.1/A.7
SGD_BASELINE_DB_APPROX = 13.83     # originally reported at 512x512; full-res SGD final values run ~13-14dB per the LR-sweep seed-check


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_neural_prior(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                                 pad_factor=PAD_FACTOR, seed=SEEDS[0], lr=LR)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    results = []
    for seed in SEEDS:
        t0 = time.time()
        _, history, _, n_params = multiplane_neural_prior(
            targets, DEPTHS_M, WAVELENGTH, DX, N_STEPS, lr=LR, device=DEVICE,
            pad_factor=PAD_FACTOR, seed=seed
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        best = max(history)
        best_step = history.index(best) + 1
        final = history[-1]
        results.append({"seed": seed, "history": history, "best": best,
                         "best_step": best_step, "final": final, "elapsed": elapsed})
        print(f"\nseed={seed}: best={best:.2f} dB (step {best_step}), final={final:.2f} dB, "
              f"{elapsed:.1f}s ({elapsed/N_STEPS:.2f}s/step)")

    bests = [r["best"] for r in results]
    finals = [r["final"] for r in results]
    best_avg = sum(bests) / len(bests)
    final_avg = sum(finals) / len(finals)

    print("\n" + "=" * 80)
    print(f"{'Seed':<8}{'Best dB':<12}{'Best step':<12}{'Final dB':<12}")
    for r in results:
        print(f"{r['seed']:<8}{r['best']:<12.2f}{r['best_step']:<12}{r['final']:<12.2f}")
    print("=" * 80)
    print(f"\nBest-ever average across 3 seeds: {best_avg:.2f} dB (range {min(bests):.2f}-{max(bests):.2f})")
    print(f"Final-value average across 3 seeds: {final_avg:.2f} dB (range {min(finals):.2f}-{max(finals):.2f})")
    print(f"\nFor comparison, this project's existing full-resolution, multi-seed baselines:")
    print(f"  GS (10.1/A.7):  11.47-12.00 dB (5-seed range, at true convergence)")
    print(f"  SGD (10.2/A.14.2 LR-sweep): ~13-14 dB region (full-res, 3-seed, at true convergence)")
    print(f"\nNeural-prior's BEST-EVER average ({best_avg:.2f} dB) is the fairest number to cite,")
    print("since its final value is actively worse due to the collapse documented in the")
    print("longrun diagnostic. Even so, compare it against GS/SGD's FINAL (converged) values --")
    print("this is not an apples-to-apples budget-matched comparison, it's 'best case for")
    print("neural-prior' vs 'standard case for the alternatives'.")

    if best_avg < GS_BASELINE_DB[0]:
        print("\nEven neural-prior's best-ever quality falls short of GS's converged full-resolution")
        print("baseline -- the network's structural bias does not help at this resolution, full stop.")
    elif best_avg < SGD_BASELINE_DB_APPROX:
        print("\nNeural-prior's best-ever quality beats GS but still falls short of SGD's converged")
        print("full-resolution quality -- no benefit from the network's structural bias at this scale,")
        print("even given the most generous possible reading (best checkpoint, not final value).")
    else:
        print("\nNeural-prior's best-ever quality matches or exceeds SGD's converged full-resolution")
        print("quality -- but only reachable via early stopping at an unpredictable, seed-dependent")
        print("point, which is a real practical cost the small-scale comparison never had to pay.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        ax.plot(range(1, N_STEPS + 1), r["history"], label=f"seed {r['seed']}", linewidth=1)
        ax.scatter([r["best_step"]], [r["best"]], marker="o", zorder=5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"Neural-prior full-resolution, 3 seeds, lr={LR} (markers = per-seed best)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("neural_prior_fullres_3seed.png", dpi=130)
    print("Saved plot: neural_prior_fullres_3seed.png")


if __name__ == "__main__":
    main()
