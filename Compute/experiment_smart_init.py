"""
Section 10.1 / Appendix A.7 found that random-phase initialization causes
genuine iteration-count variance (30-65 iterations across 5 seeds at full
resolution) despite consistent final quality -- some random starting
points have a short path to the target, others a long one, and there's
no way to know which you'll draw. This tests a different idea: what if
you don't gamble on a random draw at all?

Two deterministic (seed-independent) initializations are tested against
the existing random-phase baseline:

  A) Random phase (the current baseline, 5 seeds -- see 10.1)
  B) Zero phase (flat phase everywhere) -- naive deterministic baseline,
     included for contrast, not because it's expected to win
  C) Backprop init -- reuses math already inside this project's own GS
     loop. Every GS iteration already does: take each plane's target
     amplitude with a phase, propagate backward to the SLM plane,
     average across planes. This initialization does exactly that once,
     with FLAT phase (no phase information at all) at each target plane,
     before any GS iterations run -- a standard single-shot / conjugate
     CGH initialization, not a new or unvalidated technique. It costs
     one extra backward propagate per plane, done once, not per
     iteration.

Because B and C are deterministic, running them under different `seed`
values should produce bit-identical results -- the script verifies this
directly rather than assuming it, since a "deterministic" initializer
that turns out not to be would quietly invalidate the whole comparison.

What to look for: not just whether B or C reaches a given quality in
fewer iterations on average than A, but whether they land consistently
near A's FASTEST seeds rather than its average -- that's the actual
claim worth testing, since a deterministic initializer that's merely
"average" doesn't solve the variance problem, only a fast, ZERO-variance
one does.

Small scale first (512x512), matching this project's usual pattern.
Only worth a full-resolution, multi-seed-consistent rerun (matching
10.1's methodology bar) if this shows a real, unambiguous win here.

Run on your 3060:
    python3 experiment_smart_init.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from propagation_torch import angular_spectrum_propagate
from retrieval_torch import multiplane_gs_torch, _to_tensor_targets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)          # small scale first, matches this project's usual pattern
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_BUDGET = 60          # generous budget; small-scale plateau points are typically well under this
N_RANDOM_SEEDS = 5           # matches the 10.1 seed sweep for direct comparability
PLATEAU_HOLD = 5             # consecutive iterations required within tolerance to call it plateaued
PLATEAU_EPS_DB = 0.1


def detect_plateau(history, hold=PLATEAU_HOLD, eps=PLATEAU_EPS_DB):
    """
    Same spirit as the sustained-hold detector tested in Appendix A.7:
    requires `hold` consecutive values to sit within `eps` dB of their
    own trailing average, rather than a single threshold crossing.
    Returns the 1-indexed iteration where this first holds, or None.
    """
    for i in range(hold, len(history) + 1):
        window = history[i - hold:i]
        ref = sum(window) / hold
        if max(window) - min(window) < eps and abs(window[-1] - ref) < eps:
            return i
    return None


def init_backprop(target_planes, depths_m, wavelength, dx, device, pad_factor):
    """
    Deterministic, target-informed initialization: propagate each
    plane's target amplitude (flat/zero phase -- no phase information
    used) backward to the SLM plane, average across planes. Identical
    structure to the correction step already inside multiplane_gs_torch,
    applied once, before any iterations, instead of per-iteration with a
    GS-refined phase.
    """
    targets_amp = _to_tensor_targets(target_planes, device)
    acc = torch.zeros_like(targets_amp[0], dtype=torch.complex64)
    for target_amp, z in zip(targets_amp, depths_m):
        flat_field = target_amp.to(torch.complex64)
        back = angular_spectrum_propagate(flat_field, wavelength, dx, -z, pad_factor=pad_factor)
        acc = acc + back
    slm_field = acc / len(depths_m)
    return torch.angle(slm_field)


def run_condition(label, targets, init_phase, seed):
    counter.reset()
    t0 = time.time()
    _, history, _ = multiplane_gs_torch(
        targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS_BUDGET,
        device=DEVICE, seed=seed, pad_factor=PAD_FACTOR, init_phase=init_phase
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    plateau = detect_plateau(history)
    return {"label": label, "history": history, "plateau": plateau,
            "final": history[-1], "ffts": counter.count, "elapsed": elapsed}


def main():
    print(f"Device: {DEVICE}")
    targets = make_realistic_multiplane_target(SHAPE)

    print("\nA) Random-phase baseline, 5 seeds...")
    random_runs = [run_condition(f"random-seed-{s}", targets, None, seed=s) for s in range(N_RANDOM_SEEDS)]
    for r in random_runs:
        print(f"   seed {r['label'][-1]}: plateau={r['plateau']}, final={r['final']:.2f} dB")

    print("\nB) Zero-phase init (naive deterministic baseline)...")
    zero_phase = torch.zeros(SHAPE, device=DEVICE)
    zero_runs = [run_condition("zero-phase", targets, zero_phase, seed=s) for s in (0, 1)]
    zero_identical = zero_runs[0]["history"] == zero_runs[1]["history"]
    print(f"   plateau={zero_runs[0]['plateau']}, final={zero_runs[0]['final']:.2f} dB "
          f"(determinism check across 2 seeds: {'PASS' if zero_identical else 'FAIL -- not actually deterministic'})")

    print("\nC) Backprop init (target-informed, deterministic)...")
    backprop_phase = init_backprop(targets, DEPTHS_M, WAVELENGTH, DX, DEVICE, PAD_FACTOR)
    backprop_runs = [run_condition("backprop", targets, backprop_phase, seed=s) for s in (0, 1)]
    backprop_identical = backprop_runs[0]["history"] == backprop_runs[1]["history"]
    print(f"   plateau={backprop_runs[0]['plateau']}, final={backprop_runs[0]['final']:.2f} dB "
          f"(determinism check across 2 seeds: {'PASS' if backprop_identical else 'FAIL -- not actually deterministic'})")

    random_plateaus = [r["plateau"] for r in random_runs if r["plateau"] is not None]
    random_finals = [r["final"] for r in random_runs]
    random_min, random_max = min(random_plateaus), max(random_plateaus)
    random_avg = sum(random_plateaus) / len(random_plateaus)
    random_final_avg = sum(random_finals) / len(random_finals)

    print("\n" + "=" * 78)
    print(f"{'Condition':<22}{'Plateau iter':<16}{'Final dB':<12}{'Seed-to-seed variance'}")
    print(f"{'Random (5 seeds)':<22}{f'{random_min}-{random_max} (avg {random_avg:.0f})':<16}"
          f"{f'{random_final_avg:.2f}':<12}{'Real -- see 10.1/A.7'}")
    zero_final_str = f"{zero_runs[0]['final']:.2f}"
    backprop_final_str = f"{backprop_runs[0]['final']:.2f}"
    print(f"{'Zero-phase':<22}{str(zero_runs[0]['plateau']):<16}{zero_final_str:<12}"
          f"{'None (deterministic)' if zero_identical else 'UNEXPECTED'}")
    print(f"{'Backprop init':<22}{str(backprop_runs[0]['plateau']):<16}{backprop_final_str:<12}"
          f"{'None (deterministic)' if backprop_identical else 'UNEXPECTED'}")
    print("=" * 78)

    bp_plateau = backprop_runs[0]["plateau"]
    if bp_plateau is not None:
        print(f"\nBackprop init plateau ({bp_plateau}) vs. random range ({random_min}-{random_max}):")
        if bp_plateau <= random_min:
            print("Matches or beats random's FASTEST seed, with zero variance since it's deterministic --")
            print("this is the actual claim worth escalating to a full-resolution, methodology-matched rerun.")
        elif bp_plateau <= random_avg:
            print("Beats the random average but not the fastest seed -- a real, if partial, win: consistently")
            print("good instead of gambling on a lucky draw, even if not optimal every time.")
        else:
            print("Does not beat the random average -- the flat-phase backprop guess isn't a better starting")
            print("point than random noise for this content. Worth trying a different informed initialization")
            print("before concluding the idea doesn't work, rather than ruling it out from one variant.")
    else:
        print(f"\nBackprop init did not plateau within the {N_ITERS_BUDGET}-iteration budget -- inconclusive,")
        print("consider raising N_ITERS_BUDGET before drawing any conclusion.")

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in random_runs:
        ax.plot(range(1, N_ITERS_BUDGET + 1), r["history"], color="tab:blue", alpha=0.35, linewidth=1)
    ax.plot([], [], color="tab:blue", alpha=0.6, label="Random init (5 seeds)")
    ax.plot(range(1, N_ITERS_BUDGET + 1), zero_runs[0]["history"], color="tab:orange", linewidth=2, label="Zero-phase init")
    ax.plot(range(1, N_ITERS_BUDGET + 1), backprop_runs[0]["history"], color="tab:green", linewidth=2, label="Backprop init")
    ax.set_xlabel("GS iteration")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"Initialization strategy comparison, {SHAPE[0]}x{SHAPE[1]}, real content")
    ax.legend()
    plt.tight_layout()
    plt.savefig("smart_init_comparison.png", dpi=130)
    print("\nSaved plot: smart_init_comparison.png")


if __name__ == "__main__":
    main()
