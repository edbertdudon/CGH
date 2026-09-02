"""
First real-content test (Section 10.6, item 1): does everything found
with synthetic disc/ring/checker targets hold with content that actually
resembles what this product would show? Three content types, chosen to
be a harder generalization test than any synthetic shape tried so far:

  near:  UI navigation icon (sharp graphic content)
  mid:   translated caption text (thin strokes, small features -- known
         to be a hard case for holographic reconstruction)
  far:   procedurally generated photo-like scene (gradual shading,
         texture -- closer to natural image statistics)

Same depths, same resolution, same algorithm as the existing full-
resolution baseline -- content is the only variable changed, so results
are directly comparable to the numbers already in the document.

Does not apply the edge-taper fix from 10.4 -- that investigation is
closed as non-blocking, and re-introducing it here would add a second
variable on top of the content change this test is meant to isolate.

Run on your 3060:
    python3 experiment_realistic_content.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_counter_torch import counter
from targets import make_realistic_multiplane_target, sparse_content_bounds
from retrieval_torch import multiplane_gs_torch
from metrics import psnr as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)   # doc target resolution (N_y, N_x)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]   # same as all prior full-res runs
N_ITERS = 16

PLANE_NAMES = ["near: UI icon", "mid: caption text", "far: photo-like scene"]

# reference numbers already in the document, for direct comparison
REF_SIMULTANEOUS = [11.0, 11.7, 10.3]   # synthetic targets, full res, untapered
REF_SEQUENTIAL_DELTA_AVG = 0.0           # ~0dB average, synthetic targets


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
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    targets = make_realistic_multiplane_target(SHAPE)

    # warm-up (avoids the CUDA/cuFFT timing contamination flagged in 10.3)
    print("Warming up GPU...")
    _ = run_and_time(multiplane_gs_torch, targets, DEPTHS_M, WAVELENGTH, DX, 1,
                      device=DEVICE, pad_factor=PAD_FACTOR)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- simultaneous ----
    (phase_sim, hist_sim, cps_sim), t_sim, ffts_sim = run_and_time(
        multiplane_gs_torch, targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR
    )
    final_sim = max(cps_sim.keys())
    q_sim = [psnr_np(cps_sim[final_sim][i], targets[i]) for i in range(3)]

    # ---- sequential ----
    q_seq = []
    seq_recons = []
    for i in range(3):
        (phase_i, hist_i, cps_i), t_i, ffts_i = run_and_time(
            multiplane_gs_torch, [targets[i]], [DEPTHS_M[i]], WAVELENGTH, DX, N_ITERS,
            device=DEVICE, seed=i, pad_factor=PAD_FACTOR
        )
        final_i = max(cps_i.keys())
        q = psnr_np(cps_i[final_i][0], targets[i])
        q_seq.append(q)
        seq_recons.append(cps_i[final_i][0])

    deltas = [q_seq[i] - q_sim[i] for i in range(3)]
    avg_delta = sum(deltas) / 3

    # whole-frame PSNR on sparse content (icon, text) is dominated by
    # trivially correct background pixels and can distort the apparent
    # delta between conditions -- report masked PSNR over just each
    # sparse plane's own content region. The scene plane is NOT sparse
    # (continuous gradient fills the whole frame, no dominant empty
    # background), so whole-frame PSNR is already meaningful there and
    # isn't masked.
    masked_results = {}
    for i, name in [(0, "near: UI icon"), (1, "mid: caption text")]:
        rows, cols = sparse_content_bounds(targets[i])
        q_sim_m = psnr_np(cps_sim[final_sim][i][rows, cols], targets[i][rows, cols])
        q_seq_m = psnr_np(seq_recons[i][rows, cols], targets[i][rows, cols])
        masked_results[i] = (q_sim_m, q_seq_m)

    print("=" * 70)
    print(f"REALISTIC CONTENT, full resolution {SHAPE[1]}x{SHAPE[0]}")
    print()
    for i, name in enumerate(PLANE_NAMES):
        print(f"  {name}")
        print(f"    simultaneous: {q_sim[i]:.1f} dB  (synthetic-target reference: {REF_SIMULTANEOUS[i]:.1f} dB)")
        print(f"    sequential:   {q_seq[i]:.1f} dB  (delta: {deltas[i]:+.1f} dB)")
    print()
    print(f"  Masked PSNR (content bounding box only, not whole-frame) for the two sparse")
    print(f"  planes -- whole-frame PSNR on mostly-empty content inflates/distorts the delta:")
    for i, name in [(0, "near: UI icon"), (1, "mid: caption text")]:
        q_sim_m, q_seq_m = masked_results[i]
        print(f"    {name}: simultaneous {q_sim_m:.1f} dB, sequential {q_seq_m:.1f} dB, "
              f"delta {q_seq_m - q_sim_m:+.1f} dB  (whole-frame delta was {deltas[i]:+.1f} dB)")
    print(f"  far: photo-like scene: not masked -- content fills the whole frame, no dominant")
    print(f"    empty background, so the whole-frame delta ({deltas[2]:+.1f} dB) is already meaningful.")
    print()
    print(f"Average sequential delta: {avg_delta:+.2f} dB "
          f"(synthetic-target reference: {REF_SEQUENTIAL_DELTA_AVG:+.1f} dB)")
    print()
    print("Does the quality ceiling hold? Compare each plane's simultaneous dB above")
    print("against its synthetic-target reference -- similar range confirms the ~10-13dB")
    print("ceiling isn't specific to geometric test shapes.")
    print("Does the 'no sequential advantage' finding hold? Check whether the average")
    print("delta is still close to 0dB, or whether real content (especially the text")
    print("plane, likely the hardest case) shows a real benefit that synthetic shapes hid.")
    if DEVICE == "cuda":
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print("=" * 70)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, name in enumerate(PLANE_NAMES):
        axes[0, i].imshow(targets[i], cmap="gray")
        axes[0, i].set_title(f"Target: {name}")
        axes[0, i].axis("off")
        axes[1, i].imshow(cps_sim[final_sim][i], cmap="gray")
        axes[1, i].set_title(f"Simultaneous ({q_sim[i]:.1f} dB)")
        axes[1, i].axis("off")
        axes[2, i].imshow(seq_recons[i], cmap="gray")
        axes[2, i].set_title(f"Sequential ({q_seq[i]:.1f} dB)")
        axes[2, i].axis("off")
    plt.tight_layout()
    plt.savefig("realistic_content_results.png", dpi=130)
    print("Saved plot: realistic_content_results.png")


if __name__ == "__main__":
    main()
