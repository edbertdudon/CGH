"""
Full-resolution retest of experiment_scenecut_trials.py.

That small-scale (512x512) run reversed the original scene-cut negative-
transfer finding: with a generous, plateau-auto-detected budget instead
of a small fixed cutoff, 0/5 cut pairs (including severe content
permutations) showed negative transfer, and 4/5 beat the cold-start
average outright. But that data point is at 512x512, and this project's
own history (10.3's sequential-rendering advantage, real at 512x512,
gone at full resolution; the donor-seeding reversal earlier this
session) means small-scale results have repeatedly not survived the
jump to the real target resolution unchanged. This closes that loop the
same way: same 5 pairs (2 moderate/severe shift variants, 3 severe
content permutations), same true-convergence methodology, at
2,700x4,700.

Shift magnitudes are scaled proportionally from the 512x512 script's
250px/400px (which were ~49%/~78% of that frame's width) rather than
reused as raw pixel counts, which would be a tiny fraction of the full
frame and not a comparable severity.

N_ITERS_BUDGET raised to 200 (from 100 at small scale) -- this
project's own full-resolution runs have topped out at ~166 iterations
in worst-case donor-seeding trials, so 100 would risk under-counting
convergence here. N_RANDOM_SEEDS trimmed to 2 (from 3) to keep total
runtime tractable at this resolution.

Run on your 3060 (expect ~25-30 minutes):
    python3 experiment_scenecut_trials_fullres.py
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
SHAPE = (2700, 4700)         # full target resolution
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_BUDGET = 200          # raised from 100 -- see docstring
N_RANDOM_SEEDS = 2            # trimmed from 3 -- see docstring
PLATEAU_HOLD = 5
PLATEAU_EPS_DB = 0.1
DONOR_SEED = 0

# proportional to the 512x512 script's 250px/400px (~49%/~78% of that frame's width)
SHIFT_MODERATE_PX = round(250 / 512 * SHAPE[1])   # ~2295px
SHIFT_SEVERE_PX = round(400 / 512 * SHAPE[1])     # ~3672px


def detect_plateau(history, hold=PLATEAU_HOLD, eps=PLATEAU_EPS_DB):
    for i in range(hold, len(history) + 1):
        window = history[i - hold:i]
        if max(window) - min(window) < eps and abs(window[-1] - sum(window) / hold) < eps:
            return i
    return None


def pad_crop_shift(plane, shift_px):
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
        "S1_moderate_shift": [pad_crop_shift(p, SHIFT_MODERATE_PX) for p in base],
        "S2_severe_permuteA": [far, near, mid],
        "S3_severe_permuteB": [mid, far, near],
        "S4_severe_shift": [pad_crop_shift(p, SHIFT_SEVERE_PX) for p in base],
    }
    return scenes


def solve(scene, seed, init_phase=None):
    counter.reset()
    t0 = time.time()
    phase, history, _ = multiplane_gs_torch(
        scene, DEPTHS_M, WAVELENGTH, DX, N_ITERS_BUDGET,
        device=DEVICE, seed=seed, pad_factor=PAD_FACTOR, init_phase=init_phase,
        smooth_cutoff=True
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    plateau = detect_plateau(history)
    return {"phase": phase, "history": history, "plateau": plateau, "final": history[-1], "elapsed": elapsed}


def main():
    print(f"Device: {DEVICE}")
    print(f"Shift magnitudes: moderate={SHIFT_MODERATE_PX}px, severe={SHIFT_SEVERE_PX}px "
          f"(of {SHAPE[1]}px width)\n")
    t_start = time.time()
    base = make_realistic_multiplane_target(SHAPE)
    scenes = make_scenes(base)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(scenes["S0_reference"], DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print("Solving donor phase for each scene (fixed seed, cold-start)...")
    donors = {}
    for name, content in scenes.items():
        r = solve(content, seed=DONOR_SEED)
        donors[name] = r
        status = f"plateau={r['plateau']}" if r["plateau"] else f"DID NOT PLATEAU in {N_ITERS_BUDGET} iters"
        print(f"   {name}: {status}, final={r['final']:.2f} dB, {r['elapsed']:.1f}s")

    pairs = [
        ("S0_reference", "S1_moderate_shift", "moderate"),
        ("S0_reference", "S2_severe_permuteA", "severe"),
        ("S0_reference", "S4_severe_shift", "severe"),
        ("S1_moderate_shift", "S3_severe_permuteB", "severe"),
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

    print(f"\nCompare against the 512x512 result: 0/5 negative transfer, 4/5 beat cold average.")
    if len(negative_transfers) == 0:
        print("Confirms the small-scale finding holds at full resolution -- the original 10.5 negative-")
        print("transfer concern was a fixed-budget artifact at both scales, not a real architectural risk.")
    else:
        print("Does NOT fully confirm the small-scale finding -- negative transfer reappears at full")
        print("resolution even with true-convergence methodology. Report this divergence directly, don't")
        print("average it away with the 512x512 result.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r["pair"].replace("_", " ") for r in results]
    warm_vals = [r["warm_plateau"] if r["warm_plateau"] is not None else N_ITERS_BUDGET + 20 for r in results]
    cold_vals = [r["cold_avg"] if r["cold_avg"] is not None else N_ITERS_BUDGET + 20 for r in results]
    x = range(len(results))
    ax.bar([i - 0.2 for i in x], cold_vals, width=0.4, label="Cold-start avg", color="tab:blue")
    ax.bar([i + 0.2 for i in x], warm_vals, width=0.4, label="Warm-start (from cut)", color="tab:green")
    ax.axhline(N_ITERS_BUDGET, color="gray", linestyle=":", label=f"Budget ceiling ({N_ITERS_BUDGET})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Plateau iteration (bars at ceiling = negative transfer)")
    ax.set_title(f"Scene-cut trials, full resolution ({SHAPE[1]}x{SHAPE[0]})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("scenecut_trials_fullres_comparison.png", dpi=130)
    print("\nSaved plot: scenecut_trials_fullres_comparison.png")


if __name__ == "__main__":
    main()
