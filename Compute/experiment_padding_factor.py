"""
The last unclaimed candidate from the original list of untested compute
levers: does a lower padding factor reduce the 10.1 baseline's total
compute, not just leave the sequential-rendering finding (10.3) or the
border artifact (A.3) unaffected -- both of which only checked pad_factor
as a side question, never as a direct lever on the headline FFT/power
number.

Padding exists to suppress a wraparound artifact in FFT-based propagation
(10.4). A higher pad_factor computes a larger array per FFT -- more
compute per operation -- but 10.3 already found pad_factor 1.3 gives the
same sequential-rendering result as 2.0 at full resolution, and A.3 ruled
padding out as the border-artifact's cause at both values. Neither of
those checked whether 1.3 (or lower) changes the ACTUAL compute cost
that feeds the 10.1 baseline.

Two separate questions, because a real win needs both:
  A) Does per-FFT cost actually scale down with pad_factor? (Measured
     directly via GPU timing, same method as 10.1b -- not assumed from
     the analytic padding formula, since the real cost depends on
     implementation details this script doesn't have visibility into.)
  B) Does a lower pad_factor change how many iterations are needed to
     reach the same final quality? A pad factor that's too aggressive
     might reintroduce wraparound-like artifacts that degrade quality or
     require more iterations to compensate -- that would eat into or
     reverse any per-FFT savings.

Total compute change = (relative per-FFT cost) x (relative FFT count to
converge). Both need measuring; neither alone answers the question.

Small scale first (512x512), matching this project's usual pattern, with
multi-seed convergence checks (per the standing rule from 10.7/A.9) and
the final-value-anchored plateau detector (per 10.5/A.11 -- NOT the
retired local-window one). Only worth a full-resolution confirmation if
this shows a real, safe win here -- and given this project's history
(10.3, the donor-seeding result) of small-scale wins shrinking or
reversing at full resolution, that full-res confirmation is mandatory
before any pad_factor change goes in the document, not optional.

Run on your 3060:
    python3 experiment_padding_factor.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from propagation_torch import angular_spectrum_propagate
from retrieval_torch import multiplane_gs_torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
PAD_FACTORS = [2.0, 1.5, 1.3, 1.1]   # 2.0 is the baseline used throughout this document
N_TIMING_REPS = 30                    # repetitions for stable per-FFT wall-clock timing
N_ITERS_BUDGET = 150
N_SEEDS = 3                           # matches this project's practical full-res seed count
PLATEAU_EPS_DB = 0.1


def find_plateau_robust(history, eps=PLATEAU_EPS_DB):
    """Final-value-anchored detector -- see 10.5/A.11. Not the retired
    local-window method."""
    final = history[-1]
    for i in range(len(history)):
        if all(abs(v - final) < eps for v in history[i:]):
            return i + 1
    return len(history)


def measure_ms_per_fft(pad_factor, shape, reps=N_TIMING_REPS):
    """Direct GPU timing of a single forward propagate call at a given
    pad_factor, same method as 10.1b (precise GPU-side timing, not an
    assumed formula)."""
    field = torch.exp(1j * torch.rand(shape, device=DEVICE) * 2 * 3.14159)
    # warmup
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
    counter.reset()
    _, history, _ = multiplane_gs_torch(
        targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS_BUDGET,
        device=DEVICE, seed=seed, pad_factor=pad_factor
    )
    plateau = find_plateau_robust(history)
    return {"plateau": plateau, "final": history[-1], "ffts_at_plateau": plateau * 12}  # 4 FFT ops/plane x 3 planes


def main():
    print(f"Device: {DEVICE}")
    targets = make_realistic_multiplane_target(SHAPE)

    print("\nPART A: Per-FFT wall-clock cost at each pad_factor (direct GPU timing)...")
    ms_per_fft = {}
    for pf in PAD_FACTORS:
        ms = measure_ms_per_fft(pf, SHAPE)
        ms_per_fft[pf] = ms
        rel = ms / ms_per_fft[PAD_FACTORS[0]] if pf != PAD_FACTORS[0] else 1.0
        print(f"   pad_factor={pf}: {ms:.3f} ms/FFT" + (f" ({rel*100:.1f}% of baseline)" if pf != PAD_FACTORS[0] else " (baseline)"))

    print(f"\nPART B: Convergence behavior at each pad_factor ({N_SEEDS} seeds each, true convergence)...")
    convergence_results = {}
    for pf in PAD_FACTORS:
        print(f"\n  pad_factor={pf}:")
        runs = [solve_at_pad_factor(targets, pf, seed=s) for s in range(N_SEEDS)]
        avg_ffts = sum(r["ffts_at_plateau"] for r in runs) / len(runs)
        avg_final = sum(r["final"] for r in runs) / len(runs)
        min_ffts = min(r["ffts_at_plateau"] for r in runs)
        max_ffts = max(r["ffts_at_plateau"] for r in runs)
        convergence_results[pf] = {"avg_ffts": avg_ffts, "avg_final": avg_final, "min_ffts": min_ffts, "max_ffts": max_ffts}
        print(f"     FFTs to converge: {min_ffts}-{max_ffts} (avg {avg_ffts:.0f}), final quality: {avg_final:.2f} dB")

    baseline_pf = PAD_FACTORS[0]
    baseline_ms = ms_per_fft[baseline_pf]
    baseline_ffts = convergence_results[baseline_pf]["avg_ffts"]
    baseline_final = convergence_results[baseline_pf]["avg_final"]
    baseline_total = baseline_ms * baseline_ffts

    print("\n" + "=" * 100)
    print(f"{'pad_factor':<12}{'ms/FFT':<12}{'Avg FFTs':<12}{'Final dB':<12}{'Δ quality':<12}{'Total compute vs. baseline':<28}")
    for pf in PAD_FACTORS:
        ms = ms_per_fft[pf]
        ffts = convergence_results[pf]["avg_ffts"]
        final = convergence_results[pf]["avg_final"]
        dq = final - baseline_final
        total = ms * ffts
        rel_total = 100 * total / baseline_total
        marker = " <- baseline" if pf == baseline_pf else ""
        print(f"{pf:<12}{ms:<12.3f}{ffts:<12.0f}{final:<12.2f}{dq:<+12.2f}{rel_total:<8.1f}%{marker}")
    print("=" * 100)

    print("\nInterpretation: a real, safe win needs LOWER total compute (%) AND quality within noise (~±0.1-0.2 dB)")
    print("of baseline. A lower pad_factor that saves compute but costs meaningful quality, or needs enough")
    print("extra iterations to erase the per-FFT savings, is not a free win -- check both columns together,")
    print("not total compute alone.")

    for pf in PAD_FACTORS[1:]:
        total = ms_per_fft[pf] * convergence_results[pf]["avg_ffts"]
        rel_total = 100 * total / baseline_total
        dq = convergence_results[pf]["avg_final"] - baseline_final
        if rel_total < 90 and abs(dq) < 0.2:
            print(f"\npad_factor={pf}: real candidate -- {rel_total:.0f}% of baseline compute, quality within noise "
                  f"({dq:+.2f} dB). Worth a full-resolution confirmation before updating the 10.1 baseline.")
        elif rel_total < 90:
            print(f"\npad_factor={pf}: saves compute ({rel_total:.0f}%) but quality moved {dq:+.2f} dB -- not a free "
                  f"win without checking whether that's an acceptable tradeoff.")

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
    plt.title(f"Padding factor: compute vs. quality tradeoff, {SHAPE[0]}x{SHAPE[1]}")
    plt.tight_layout()
    plt.savefig("padding_factor_comparison.png", dpi=130)
    print("\nSaved plot: padding_factor_comparison.png")


if __name__ == "__main__":
    main()
