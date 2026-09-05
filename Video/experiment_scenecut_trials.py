"""
10.5 tested exactly one scene-cut case and found negative transfer at
512x512 (warm-start never converged) that didn't replicate at full
resolution (converged in 2 iterations). That's a sample size of one, at
each scale, and per 10.7/A.9's newly-established standing rule, neither
of those numbers was checked against TRUE, plateau-verified convergence
-- they used a small fixed iteration budget, exactly the setup that
flipped the donor-seeding result from a 64% win to a real loss elsewhere
in this project. This script fixes both problems at once: more cut
pairs, and true-convergence methodology applied from the start rather
than retrofitted after the fact.

Five scene-cut pairs are constructed at two severities:
  - MODERATE cuts: a pad-crop shift of the same base content (partial
    overlap with the previous frame, some genuinely new/blank region
    introduced -- non-wrapping, unlike torch.roll, per the earlier
    wraparound investigation)
  - SEVERE cuts: a permutation of which content (icon/text/photo) sits
    at which depth plane -- genuinely unrelated content per plane, not
    just spatially shifted

For each pair (A -> B):
  - A fixed donor phase is solved for scene A (cold-start, seed 0)
  - Scene B is cold-started from 3 random seeds (baseline, capturing the
    seed variance documented in 10.1/A.7 rather than a single point)
  - Scene B is warm-started once from A's converged phase (deterministic
    -- no seed dependence once a fixed donor phase is supplied)
  - ALL of the above run to a generous, plateau-auto-detected budget,
    not a small fixed cutoff -- see detect_plateau()

Small scale first (512x512), matching this project's usual pattern.
Only worth a full-resolution rerun if a consistent pattern emerges here
across multiple pairs -- a single full-res data point already exists
from 10.5 but was measured under the now-retired fixed-budget standard.

Run on your 3060:
    python3 experiment_scenecut_trials.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)          # small scale first, matches this project's usual pattern
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_BUDGET = 100         # generous; prior small-scale plateaus have topped out well under this
N_RANDOM_SEEDS = 3           # cold-start baseline per pair, capturing seed variance per 10.1/A.7
PLATEAU_HOLD = 5
PLATEAU_EPS_DB = 0.1
DONOR_SEED = 0                # fixed seed used to solve every "A" scene, for reproducibility


def detect_plateau(history, hold=PLATEAU_HOLD, eps=PLATEAU_EPS_DB):
    """Same sustained-hold criterion validated in Appendix A.7/A.9. Returns
    1-indexed plateau iteration, or None if never plateaued in-budget."""
    for i in range(hold, len(history) + 1):
        window = history[i - hold:i]
        if max(window) - min(window) < eps and abs(window[-1] - sum(window) / hold) < eps:
            return i
    return None


def pad_crop_shift(plane, shift_px):
    """Non-wrapping horizontal shift: content shifted off one edge is
    genuinely lost, not wrapped to the other side (torch.roll's
    wraparound was investigated and left unexplained in 10.6/A.3 --
    avoided here on purpose, not because it was ever confirmed as the
    cause, but because it's not worth reopening in this experiment)."""
    t = torch.as_tensor(plane)
    h, w = t.shape
    shifted = torch.zeros_like(t)
    if shift_px == 0:
        return plane
    elif shift_px > 0:
        shifted[:, shift_px:] = t[:, :w - shift_px]
    else:
        s = -shift_px
        shifted[:, :w - s] = t[:, s:]
    return shifted.numpy()


def make_scenes(base):
    near, mid, far = base
    scenes = {
        "S0_reference": [near, mid, far],
        "S1_moderate_shift250": [pad_crop_shift(p, 250) for p in base],
        "S2_severe_permuteA": [far, near, mid],
        "S3_severe_permuteB": [mid, far, near],
        "S4_severe_shift400": [pad_crop_shift(p, 400) for p in base],
    }
    return scenes


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
    return {"phase": phase, "history": history, "plateau": plateau, "final": history[-1], "elapsed": elapsed}


def main():
    print(f"Device: {DEVICE}")
    base = make_realistic_multiplane_target(SHAPE)
    scenes = make_scenes(base)

    # Solve a fixed donor phase for every scene once (seed=DONOR_SEED), so any
    # scene can act as the "previous frame" (A) in a cut pair.
    print("Solving donor phase for each scene (fixed seed, cold-start)...")
    donors = {}
    for name, content in scenes.items():
        r = solve(content, seed=DONOR_SEED)
        donors[name] = r
        status = f"plateau={r['plateau']}" if r["plateau"] else f"DID NOT PLATEAU in {N_ITERS_BUDGET} iters"
        print(f"   {name}: {status}, final={r['final']:.2f} dB")

    pairs = [
        ("S0_reference", "S1_moderate_shift250", "moderate"),
        ("S0_reference", "S2_severe_permuteA", "severe"),
        ("S0_reference", "S4_severe_shift400", "severe"),
        ("S1_moderate_shift250", "S3_severe_permuteB", "severe"),
        ("S2_severe_permuteA", "S0_reference", "severe"),
    ]

    print(f"\nRunning {len(pairs)} scene-cut trials (cold baseline: {N_RANDOM_SEEDS} seeds each)...\n")
    results = []
    for a_name, b_name, severity in pairs:
        print(f"Cut: {a_name} -> {b_name} ({severity})")
        cold_runs = [solve(scenes[b_name], seed=s) for s in range(N_RANDOM_SEEDS)]
        cold_plateaus = [r["plateau"] for r in cold_runs if r["plateau"] is not None]
        cold_avg = sum(cold_plateaus) / len(cold_plateaus) if cold_plateaus else None
        cold_min = min(cold_plateaus) if cold_plateaus else None
        cold_max = max(cold_plateaus) if cold_plateaus else None

        warm = solve(scenes[b_name], seed=0, init_phase=donors[a_name]["phase"])
        warm_status = f"{warm['plateau']}" if warm["plateau"] is not None else f"NEGATIVE TRANSFER (no plateau in {N_ITERS_BUDGET})"

        cold_str = f"{cold_min}-{cold_max} (avg {cold_avg:.1f})" if cold_avg is not None else "did not plateau"
        print(f"   cold: {cold_str} | warm: {warm_status}")

        results.append({
            "pair": f"{a_name} -> {b_name}", "severity": severity,
            "cold_avg": cold_avg, "cold_min": cold_min, "cold_max": cold_max,
            "warm_plateau": warm["plateau"], "warm_final": warm["final"],
            "cold_final_avg": sum(r["final"] for r in cold_runs) / len(cold_runs),
        })

    print("\n" + "=" * 90)
    print(f"{'Pair':<38}{'Severity':<10}{'Cold range (avg)':<22}{'Warm':<20}")
    for r in results:
        cold_disp = f"{r['cold_min']}-{r['cold_max']} ({r['cold_avg']:.1f})" if r["cold_avg"] is not None else "n/a"
        warm_disp = str(r["warm_plateau"]) if r["warm_plateau"] is not None else "NEGATIVE TRANSFER"
        print(f"{r['pair']:<38}{r['severity']:<10}{cold_disp:<22}{warm_disp:<20}")
    print("=" * 90)

    negative_transfers = [r for r in results if r["warm_plateau"] is None]
    helped = [r for r in results if r["warm_plateau"] is not None and r["cold_avg"] is not None
              and r["warm_plateau"] < r["cold_avg"]]
    hurt = [r for r in results if r["warm_plateau"] is not None and r["cold_avg"] is not None
            and r["warm_plateau"] >= r["cold_avg"]]

    print(f"\n{len(negative_transfers)}/{len(results)} pairs showed negative transfer (warm-start never plateaued).")
    print(f"{len(helped)}/{len(results)} pairs where warm-start beat the cold average.")
    print(f"{len(hurt)}/{len(results)} pairs where warm-start matched or lagged the cold average (but still converged).")

    moderate = [r for r in results if r["severity"] == "moderate"]
    severe = [r for r in results if r["severity"] == "severe"]
    print(f"\nModerate cuts: {sum(1 for r in moderate if r['warm_plateau'] is None)}/{len(moderate)} negative transfer.")
    print(f"Severe cuts:   {sum(1 for r in severe if r['warm_plateau'] is None)}/{len(severe)} negative transfer.")
    print("If negative transfer concentrates in severe cuts specifically, that's a real, actionable signal for")
    print("a cut-detection threshold (10.8) -- if it's scattered regardless of severity, the risk looks harder")
    print("to predict from content difference alone and may need a more conservative universal fallback instead.")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r["pair"].replace("_", " ") for r in results]
    warm_vals = [r["warm_plateau"] if r["warm_plateau"] is not None else N_ITERS_BUDGET + 10 for r in results]
    cold_vals = [r["cold_avg"] if r["cold_avg"] is not None else N_ITERS_BUDGET + 10 for r in results]
    x = range(len(results))
    ax.bar([i - 0.2 for i in x], cold_vals, width=0.4, label="Cold-start avg", color="tab:blue")
    ax.bar([i + 0.2 for i in x], warm_vals, width=0.4, label="Warm-start (from cut)", color="tab:green")
    ax.axhline(N_ITERS_BUDGET, color="gray", linestyle=":", label=f"Budget ceiling ({N_ITERS_BUDGET})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Plateau iteration (bars at ceiling = negative transfer)")
    ax.set_title(f"Scene-cut trials: cold vs. warm-start plateau, {SHAPE[0]}x{SHAPE[1]}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("scenecut_trials_comparison.png", dpi=130)
    print("\nSaved plot: scenecut_trials_comparison.png")


if __name__ == "__main__":
    main()
