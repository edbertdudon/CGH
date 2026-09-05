"""
Shared plateau detector for convergence sweeps.

The original detector (duplicated across experiment_end_to_end_compute.py,
experiment_sequential_vs_simultaneous_converged.py,
experiment_int8_precision_fullres_converged.py) had two compounding
flaws, found via multi-seed stress testing during the temporal
warm-start / INT8 follow-up work:

  1. It compared every point to history[-1] -- a single, itself-noisy
     sample -- as the reference "converged" value, instead of a stable
     estimate of where the curve has actually settled.
  2. It used FIRST TOUCH (first iteration >= reference - tol), so one
     lucky upward noise wobble early in a still-noisy region could
     satisfy the threshold without the algorithm having actually
     stopped improving.

Both together made the detector highly sensitive to exactly which
random seed a run happened to use: on the SAME resolution/content/
algorithm, re-running with different seeds swung the detected plateau
iteration by 28-35 iterations (e.g. 30 to 65 out of 100) even though the
underlying final quality was stable across those same seeds (dB spread
under 0.6). Any FFT/power/compute figure derived from that iteration
count inherited the same instability -- confirmed directly: the
10.1c end-to-end compute figure swung 130-282mW depending purely on
seed, using the old detector.

find_plateau_robust() fixes both: the reference value is the AVERAGE of
the last `hold` iterations (stable, not one noisy sample), and the
detected plateau iteration is the first point after which EVERY
remaining value stays within tolerance of that reference (a sustained
hold, not a first touch) -- a single early wobble can no longer trigger
a false-early detection.
"""


def find_plateau(history, tol=0.2):
    """
    ORIGINAL (unstable) detector -- kept only for direct before/after
    comparison against find_plateau_robust() below. Do not use for new
    work; see module docstring for why.
    """
    final = history[-1]
    plateau_iter = next(i + 1 for i, v in enumerate(history) if v >= final - tol)
    still_rising = len(history) >= 10 and (final - history[-10]) > tol
    return plateau_iter, final, still_rising


def find_plateau_robust(history, tol=0.2, hold=10):
    """
    Returns (plateau_iter, reference_quality, still_rising).

    reference_quality: average of the last `hold` iterations -- a stable
    estimate of converged quality, not a single noisy sample.

    plateau_iter: the first (1-indexed) iteration after which EVERY
    subsequent value stays within `tol` of reference_quality. This is a
    sustained-hold criterion, not first-touch -- a single early noise
    spike that happens to graze the band does not count unless the
    curve actually stays there from that point on.

    still_rising: True if the reference (last `hold` iterations) is
    still meaningfully above the PRIOR `hold`-iteration block, i.e. the
    curve was still improving at a rate above `tol` as of the end of the
    run -- a signal to extend N_ITERS_MAX before trusting the result.
    """
    n = len(history)
    hold = min(hold, n)
    reference = sum(history[-hold:]) / hold

    plateau_iter = n
    for i in range(n):
        if all(v >= reference - tol for v in history[i:]):
            plateau_iter = i + 1
            break

    still_rising = False
    if n >= 2 * hold:
        prior_block = sum(history[-2 * hold:-hold]) / hold
        still_rising = (reference - prior_block) > tol

    return plateau_iter, reference, still_rising
