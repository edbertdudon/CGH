"""
Two things in one script:

  1. Confirm the edge-taper fix (proposed in Section 10.4) actually
     removes the border grid artifact, at 2700x2700 where you already
     have a clean before/after baseline to compare against.

  2. Re-run the full-resolution (4,700x2,700) sequential-vs-simultaneous
     comparison WITH the fix applied. This is the direct confirmation
     Section 10.6 called for -- until now, "the artifact probably
     doesn't bias the delta" was inference (common-mode reasoning), not
     a measurement. This makes it a measurement.

Note on what the taper represents physically: forcing SLM amplitude down
at the edges is not strictly "phase-only" anymore in the idealized
sense, but it's a reasonable stand-in for something real displays
actually have -- illumination is rarely perfectly uniform across the
panel (laser beams are commonly closer to Gaussian than top-hat). Worth
flagging to an optical engineer as a modeling simplification either way,
not asserting it as confirmed device physics.

Run on your 3060:
    python3 experiment_edge_taper_fix.py
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_multiplane_target
from retrieval_torch import multiplane_gs_torch
from metrics import psnr as psnr_np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS = 16
PAD_FACTOR = 2.0
TAPER_WIDTH = 0.08


def main():
    print(f"Device: {DEVICE}")

    # ---- Part 1: does the taper remove the artifact? (2700x2700, fast) ----
    print("Part 1: taper on vs off at 2700x2700...")
    diag_shape = (2700, 2700)
    diag_targets = make_multiplane_target(diag_shape, n_planes=3, soft=True, sigma=8.0)

    _, _, cps_notaper = multiplane_gs_torch(
        diag_targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR,
        edge_taper=False
    )
    final_nt = max(cps_notaper.keys())
    q_notaper = [psnr_np(cps_notaper[final_nt][i], diag_targets[i]) for i in range(3)]

    _, _, cps_taper = multiplane_gs_torch(
        diag_targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR,
        edge_taper=True, taper_width=TAPER_WIDTH
    )
    final_t = max(cps_taper.keys())
    q_taper = [psnr_np(cps_taper[final_t][i], diag_targets[i]) for i in range(3)]

    print(f"  no taper quality: {[round(q,1) for q in q_notaper]}")
    print(f"  taper quality:    {[round(q,1) for q in q_taper]}")

    fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        axes1[0, i].imshow(cps_notaper[final_nt][i], cmap="gray")
        axes1[0, i].set_title(f"no taper, plane {i+1} ({q_notaper[i]:.1f} dB)")
        axes1[0, i].axis("off")
        axes1[1, i].imshow(cps_taper[final_t][i], cmap="gray")
        axes1[1, i].set_title(f"edge taper, plane {i+1} ({q_taper[i]:.1f} dB)")
        axes1[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("edge_taper_diagnostic.png", dpi=130)
    print("  Saved: edge_taper_diagnostic.png")
    print()

    # ---- Part 2: confirming re-test at full resolution, with the fix applied ----
    print("Part 2: full-resolution (4700x2700) sequential vs simultaneous, WITH taper...")
    full_shape = (2700, 4700)
    full_targets = make_multiplane_target(full_shape, n_planes=3, soft=True, sigma=8.0)

    _, _, cps_sim = multiplane_gs_torch(
        full_targets, DEPTHS_M, WAVELENGTH, DX, N_ITERS, device=DEVICE, pad_factor=PAD_FACTOR,
        edge_taper=True, taper_width=TAPER_WIDTH
    )
    final_sim = max(cps_sim.keys())
    q_sim = [psnr_np(cps_sim[final_sim][i], full_targets[i]) for i in range(3)]

    q_seq = []
    seq_recons = []
    for i in range(3):
        _, _, cps_i = multiplane_gs_torch(
            [full_targets[i]], [DEPTHS_M[i]], WAVELENGTH, DX, N_ITERS, device=DEVICE, seed=i,
            pad_factor=PAD_FACTOR, edge_taper=True, taper_width=TAPER_WIDTH
        )
        final_i = max(cps_i.keys())
        q = psnr_np(cps_i[final_i][0], full_targets[i])
        q_seq.append(q)
        seq_recons.append(cps_i[final_i][0])

    deltas = [q_seq[i] - q_sim[i] for i in range(3)]
    avg_delta = sum(deltas) / 3

    print("=" * 70)
    print(f"WITH edge taper (width={TAPER_WIDTH}), full resolution {full_shape[1]}x{full_shape[0]}:")
    for i in range(3):
        print(f"  plane {i+1}: simultaneous {q_sim[i]:.1f} dB, sequential {q_seq[i]:.1f} dB, "
              f"delta {deltas[i]:+.1f} dB")
    print(f"  average delta: {avg_delta:+.2f} dB")
    print()
    print("Compare against the untapered full-resolution result already in the document:")
    print("  deltas were -0.4, +0.5, +0.1 dB (avg ~0.0 dB) without the taper.")
    print("If this run lands in the same range, the ~0dB finding is now directly confirmed,")
    print("not just inferred from common-mode reasoning. If it differs meaningfully, the")
    print("artifact WAS biasing the comparison and this run supersedes the untapered one.")
    print("=" * 70)

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        axes2[0, i].imshow(cps_sim[final_sim][i], cmap="gray")
        axes2[0, i].set_title(f"Simultaneous, plane {i+1} ({q_sim[i]:.1f} dB)")
        axes2[0, i].axis("off")
        axes2[1, i].imshow(seq_recons[i], cmap="gray")
        axes2[1, i].set_title(f"Sequential, plane {i+1} ({q_seq[i]:.1f} dB)")
        axes2[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("full_resolution_taper_confirmed.png", dpi=130)
    print("Saved: full_resolution_taper_confirmed.png")


if __name__ == "__main__":
    main()
