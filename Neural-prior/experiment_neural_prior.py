"""
The literature check found GS is explicitly the outdated baseline in
current CGH research -- neural and hybrid approaches are the real
frontier. This is the first, cheapest test of that idea: not a full
trained, generalizable network (expensive: needs a dataset, training
time, and carries a real generalization risk per the literature) but a
network FIT TO THIS ONE SCENE, the same way SGD fits pixels to this one
scene. No training dataset needed. If a convolutional network's
structural bias reaches a given quality with fewer FFT calls than raw
pixel-wise SGD, that's real evidence a smarter algorithm could reduce
the 696-FFT/frame figure in Section 10.1c -- if it doesn't help, that's
equally real evidence worth knowing before investing in a full trained
network.

Three conditions, identical target, identical propagation model:
  A) GS (the baseline this whole document has been measuring)
  B) SGD (direct pixel-wise gradient descent -- already tested)
  C) Neural-prior (CNN-parameterized phase, gradient descent on the
     network's weights instead of the pixels directly)

Small scale first (512x512) for fast iteration -- same discipline as
every other algorithm comparison in this project. Only worth scaling to
full resolution if this shows a real difference here.

Run on your 3060:
    python3 experiment_neural_prior.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target
from retrieval_torch import multiplane_gs_torch, multiplane_sgd, multiplane_neural_prior
from metrics import psnr as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)          # small scale first, matches this project's usual pattern
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_GS = 16
N_STEPS_SGD = 500            # matches the budget SGD needed to look competitive earlier
N_STEPS_NEURAL = 500         # same step budget as SGD, for a fair comparison
PLANE_NAMES = ["near", "mid", "far"]


def run_and_time(fn, *args, **kwargs):
    counter.reset()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    result = fn(*args, **kwargs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return result, elapsed, counter.count


def main():
    print(f"Device: {DEVICE}")
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE, pad_factor=PAD_FACTOR)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print("\nA) GS...")
    (phase_gs, hist_gs, cps_gs), t_gs, ffts_gs = run_and_time(
        multiplane_gs_torch, targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS_GS, device=DEVICE, pad_factor=PAD_FACTOR
    )
    print(f"   final: {hist_gs[-1]:.2f} dB, {ffts_gs} FFTs, {t_gs:.2f}s")

    print("\nB) SGD (direct pixel optimization)...")
    (phase_sgd, hist_sgd, cps_sgd, _), t_sgd, ffts_sgd = run_and_time(
        multiplane_sgd, targets, DEPTHS_M, WAVELENGTH, DX, N_STEPS_SGD, device=DEVICE
    )
    print(f"   final: {hist_sgd[-1]:.2f} dB, {ffts_sgd} FFTs, {t_sgd:.2f}s")

    print("\nC) Neural-prior (CNN-parameterized phase, fit to this scene)...")
    (phase_neural, hist_neural, cps_neural, n_params), t_neural, ffts_neural = run_and_time(
        multiplane_neural_prior, targets, DEPTHS_M, WAVELENGTH, DX, N_STEPS_NEURAL, device=DEVICE
    )
    print(f"   final: {hist_neural[-1]:.2f} dB, {ffts_neural} FFTs, {t_neural:.2f}s, {n_params:,} network params")

    print("\n" + "=" * 70)
    print(f"{'Method':<12}{'Final dB':<12}{'FFT calls':<12}{'Wall-clock':<12}")
    print(f"{'GS':<12}{hist_gs[-1]:<12.2f}{ffts_gs:<12}{t_gs:<12.2f}")
    print(f"{'SGD':<12}{hist_sgd[-1]:<12.2f}{ffts_sgd:<12}{t_sgd:<12.2f}")
    print(f"{'Neural':<12}{hist_neural[-1]:<12.2f}{ffts_neural:<12}{t_neural:<12.2f}")
    print("=" * 70)
    print("Note: SGD and Neural FFT counts are forward-pass only -- both also run backward()")
    print("internally, which does real, invisible compute the counter can't see (same caveat")
    print("as every SGD comparison in this project). Neural additionally pays for the CNN's")
    print("own forward+backward cost, not counted here at all -- wall-clock time is the more")
    print("honest comparison between B and C than FFT count is.")
    if hist_neural[-1] > hist_sgd[-1] + 0.5:
        print("\nNeural-prior beats plain SGD at equal step budget -- the structural bias is")
        print("doing real work. Worth scaling to full resolution and testing a real training")
        print("set next.")
    elif hist_neural[-1] < hist_sgd[-1] - 0.5:
        print("\nNeural-prior underperforms plain SGD here -- the network architecture or")
        print("learning rate likely needs tuning before drawing a conclusion either way.")
    else:
        print("\nRoughly comparable to plain SGD at this scale -- the real test is whether a")
        print("network trained across MANY scenes (not just fit to one) generalizes well")
        print("enough to skip iteration entirely at inference time. That's the next, bigger")
        print("experiment, not this one.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, N_ITERS_GS + 1), hist_gs, label="GS", marker="o")
    ax.plot(range(1, N_STEPS_SGD + 1), hist_sgd, label="SGD (pixels)", linewidth=1)
    ax.plot(range(1, N_STEPS_NEURAL + 1), hist_neural, label="Neural-prior (CNN weights)", linewidth=1)
    ax.set_xlabel("Iteration / step")
    ax.set_ylabel("Mean PSNR across planes (dB)")
    ax.set_title(f"GS vs SGD vs Neural-prior, {SHAPE[0]}x{SHAPE[1]}, real content")
    ax.legend()
    plt.tight_layout()
    plt.savefig("neural_prior_comparison.png", dpi=130)
    print("Saved plot: neural_prior_comparison.png")

    final_neural = max(cps_neural.keys())
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, name in enumerate(PLANE_NAMES):
        axes[i].imshow(cps_neural[final_neural][i], cmap="gray")
        axes[i].set_title(f"Neural-prior, {name} ({psnr_np(cps_neural[final_neural][i], targets[i]):.1f} dB)")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig("neural_prior_recon.png", dpi=130)
    print("Saved plot: neural_prior_recon.png")


if __name__ == "__main__":
    main()
