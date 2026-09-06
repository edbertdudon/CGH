"""
Full-resolution, multi-seed confirmation of experiment_padding_factor.py's
512x512 result, which is NOT optional before trusting it: every lower
pad_factor tested (1.5, 1.3, 1.1) came out BOTH cheaper (57-75% of
baseline total compute) AND higher quality (+0.52 to +1.16 dB) than the
pad_factor=2.0 baseline used throughout this document -- a strict win on
both axes, not a tradeoff. That's counter to padding's stated purpose
(suppressing a wraparound artifact -- 10.4/A.3), and this project has
repeatedly found small-scale wins shrink or reverse at full resolution
(10.3's sequential-rendering finding, A.9's donor-seeding result, A.13's
two-plane-dropoff retraction) -- so this is treated as a hypothesis to
attack, not a result to adopt, exactly as experiment_padding_factor.py's
own docstring requires.

Same two-part methodology, same pad_factor set, now at full resolution
(4,700x2,700) with 3 seeds per condition and a generous, plateau-verified
convergence budget (still_rising checked explicitly, not assumed).

Run on your 3060 (expect ~40-50 min: 4 pad_factors x 3 seeds, full res):
    python3 experiment_padding_factor_fullres.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target
from propagation_torch import angular_spectrum_propagate
from retrieval_torch import multiplane_gs_torch
from convergence import find_plateau_robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
PAD_FACTORS = [2.0, 1.5, 1.3, 1.1]   # 2.0 is the baseline used throughout this document
N_TIMING_REPS = 20
N_ITERS_BUDGET = 300
N_SEEDS = 3


def measure_ms_per_fft(pad_factor, shape, reps=N_TIMING_REPS):
    field = torch.exp(1j * torch.rand(shape, device=DEVICE) * 2 * 3.14159)
    for _ in range(3):
        _ = angular_spectrum_propagate(field, WAVELENGTH, DX, DEPTHS_M[0], pad_factor=pad_factor)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            _ = angular_spectrum_propagate(field, WAVELENGTH, DX, DEPTHS_M[0], pad_factor=pad_factor)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps
    else:
        t0 = time.time()
        for _ in range(reps):
            _ = angular_spectrum_propagate(field, WAVELENGTH, DX, DEPTHS_M[0], pad_factor=pad_factor)
        return (time.time() - t0) * 1000 / reps


def solve_at_pad_factor(targets, pad_factor, seed):
    t0 = time.time()
    _, history, _ = multiplane_gs_torch(
        targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS_BUDGET,
        device=DEVICE, seed=seed, pad_factor=pad_factor, smooth_cutoff=True
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    plateau, reference, still_rising = find_plateau_robust(history)
    return {"plateau": plateau, "final": history[-1], "ffts_at_plateau": plateau * 12,
            "still_rising": still_rising, "elapsed": elapsed}


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE, pad_factor=2.0)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print("\nPART A: Per-FFT wall-clock cost at each pad_factor (direct GPU timing, full resolution)...")
    ms_per_fft = {}
    for pf in PAD_FACTORS:
        ms = measure_ms_per_fft(pf, SHAPE)
        ms_per_fft[pf] = ms
        rel = ms / ms_per_fft[PAD_FACTORS[0]] if pf != PAD_FACTORS[0] else 1.0
        print(f"   pad_factor={pf}: {ms:.3f} ms/FFT" + (f" ({rel*100:.1f}% of baseline)" if pf != PAD_FACTORS[0] else " (baseline)"))

    print(f"\nPART B: Convergence behavior at each pad_factor ({N_SEEDS} seeds each, full resolution, true convergence)...")
    convergence_results = {}
    for pf in PAD_FACTORS:
        print(f"\n  pad_factor={pf}:")
        runs = []
        for s in range(N_SEEDS):
            r = solve_at_pad_factor(targets, pf, seed=s)
            runs.append(r)
            print(f"     seed={s}: plateau={r['plateau']}, final={r['final']:.2f} dB, {r['elapsed']:.1f}s"
                  f"{' STILL RISING -- N_ITERS may be too low' if r['still_rising'] else ''}")
        avg_ffts = sum(r["ffts_at_plateau"] for r in runs) / len(runs)
        avg_final = sum(r["final"] for r in runs) / len(runs)
        min_ffts = min(r["ffts_at_plateau"] for r in runs)
        max_ffts = max(r["ffts_at_plateau"] for r in runs)
        min_final = min(r["final"] for r in runs)
        max_final = max(r["final"] for r in runs)
        convergence_results[pf] = {"avg_ffts": avg_ffts, "avg_final": avg_final,
                                     "min_ffts": min_ffts, "max_ffts": max_ffts,
                                     "min_final": min_final, "max_final": max_final}
        print(f"     FFTs to converge: {min_ffts}-{max_ffts} (avg {avg_ffts:.0f}), "
              f"final quality: {min_final:.2f}-{max_final:.2f} dB (avg {avg_final:.2f})")

    baseline_pf = PAD_FACTORS[0]
    baseline_ms = ms_per_fft[baseline_pf]
    baseline_ffts = convergence_results[baseline_pf]["avg_ffts"]
    baseline_final = convergence_results[baseline_pf]["avg_final"]
    baseline_total = baseline_ms * baseline_ffts

    print("\n" + "=" * 100)
    print(f"{'pad_factor':<12}{'ms/FFT':<12}{'Avg FFTs':<12}{'Final dB':<12}{'Delta quality':<14}{'Total compute vs. baseline'}")
    for pf in PAD_FACTORS:
        ms = ms_per_fft[pf]
        ffts = convergence_results[pf]["avg_ffts"]
        final = convergence_results[pf]["avg_final"]
        dq = final - baseline_final
        total = ms * ffts
        rel_total = 100 * total / baseline_total
        marker = " <- baseline" if pf == baseline_pf else ""
        print(f"{pf:<12}{ms:<12.3f}{ffts:<12.0f}{final:<12.2f}{dq:<+14.2f}{rel_total:.1f}%{marker}")
    print("=" * 100)

    print("\nInterpretation: the 512x512 result showed every lower pad_factor as a strict win on")
    print("both compute and quality -- surprising given padding exists to suppress an artifact, and")
    print("counter to this project's usual small-scale-to-full-resolution pattern (wins shrinking or")
    print("reversing). Checking whether that holds here:")

    any_win = False
    any_reversed = False
    for pf in PAD_FACTORS[1:]:
        total = ms_per_fft[pf] * convergence_results[pf]["avg_ffts"]
        rel_total = 100 * total / baseline_total
        dq = convergence_results[pf]["avg_final"] - baseline_final
        if rel_total < 90 and dq > -0.2:
            print(f"\npad_factor={pf}: STILL a win at full resolution -- {rel_total:.0f}% of baseline compute, "
                  f"quality delta {dq:+.2f} dB.")
            any_win = True
        elif rel_total >= 90:
            print(f"\npad_factor={pf}: compute saving did NOT hold at full resolution ({rel_total:.0f}% of baseline,"
                  f" expected <90%) -- the 512x512 result does not transfer.")
            any_reversed = True
        elif dq < -0.2:
            print(f"\npad_factor={pf}: compute still saves ({rel_total:.0f}%) but quality REVERSED to a real loss"
                  f" ({dq:+.2f} dB) at full resolution -- the 512x512 quality-improvement result does not transfer.")
            any_reversed = True

    if any_win and not any_reversed:
        print("\nThe small-scale result holds at full resolution: lower pad_factor is a genuine free win here.")
        print("Worth updating the 10.1 baseline and Section 4.4/4.5 derivations to reflect this.")
    elif any_reversed:
        print("\nThe small-scale result does NOT fully transfer to full resolution -- report the full-resolution")
        print("numbers as the authoritative ones, per this project's standing rule, not the 512x512 preview.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    totals = [100 * ms_per_fft[pf] * convergence_results[pf]["avg_ffts"] / baseline_total for pf in PAD_FACTORS]
    finals = [convergence_results[pf]["avg_final"] for pf in PAD_FACTORS]
    ax1.bar([str(pf) for pf in PAD_FACTORS], totals, color="tab:blue", alpha=0.7)
    ax1.set_xlabel("pad_factor")
    ax1.set_ylabel("Total compute (% of pad_factor=2.0 baseline)", color="tab:blue")
    ax1.axhline(100, color="gray", linestyle=":")
    ax2 = ax1.twinx()
    ax2.plot([str(pf) for pf in PAD_FACTORS], finals, color="tab:red", marker="o")
    ax2.set_ylabel("Final quality (dB)", color="tab:red")
    plt.title(f"Padding factor: compute vs. quality tradeoff, full resolution ({SHAPE[1]}x{SHAPE[0]}), {N_SEEDS} seeds")
    plt.tight_layout()
    plt.savefig("padding_factor_fullres_comparison.png", dpi=130)
    print("Saved plot: padding_factor_fullres_comparison.png")


if __name__ == "__main__":
    main()
