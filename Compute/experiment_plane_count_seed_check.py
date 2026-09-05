"""
Last item flagged in the v20/v21/v22 audit: 10.2's plane-count figures
("Quality degrades monotonically with simultaneous depth-plane count:
14.6 dB (1 plane) -> 11.9 dB (2 planes) -> 11.5 dB (3 planes), corrected
metric") were never independently found in this project's committed
history -- no script producing them exists, single-seed or otherwise --
and 10.9 has flagged them since v19 as remaining single-seed and
unaudited against the multi-seed/full-resolution standard everything
else in this document now meets.

This regenerates the comparison directly at target resolution
(4,700x2,700), across 3 seeds per plane-count condition, with true
plateau-verified convergence rather than a fixed iteration budget (the
same standing rule from 10.7/10.9 applied here). Conditions:
  1 plane:  near only
  2 planes: near + mid
  3 planes: near + mid + far (this project's standard 3-plane baseline)

Final quality is reported on the corrected (intensity) metric -- see
Appendix A.2 -- not GS's internal amplitude-vs-amplitude convergence
metric, which is used only to detect the plateau, matching the
methodology caveat already flagged for 10.3's re-audit (A.13).

A first pass at this used whole-frame PSNR for every plane and got a
wildly non-monotonic, hard-to-believe result (2-plane scoring higher
than both 1-plane and 3-plane). Caught before trusting it: near (icon)
and mid (caption text) are both sparse content -- the same
whole-frame-PSNR trap already found and fixed elsewhere in this project
(Appendix A.6, 10.8) -- whole-frame PSNR on them is dominated by
trivially-correct background pixels. This version masks near and mid to
their own content bounding box (sparse_content_bounds) before scoring;
far (photo-like, fills the frame) is scored whole-frame as before.

Run on your 3060 (expect ~30-35 min):
    python3 experiment_plane_count_seed_check.py
"""
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from targets import make_realistic_multiplane_target, sparse_content_bounds
from retrieval_torch import multiplane_gs_torch
from metrics import psnr_intensity
from convergence import find_plateau_robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (2700, 4700)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_ITERS_BUDGET = 300
SEEDS = [0, 1, 2]

CONDITIONS = {
    "1 plane (near)": [0],
    "2 planes (near+mid)": [0, 1],
    "3 planes (near+mid+far)": [0, 1, 2],
}
SPARSE_PLANE_IDXS = {0, 1}  # near (icon), mid (caption text) -- far (idx 2) is dense


def solve(targets_subset, depths_subset, plane_idxs, seed):
    t0 = time.time()
    phase, history, recon_by_checkpoint = multiplane_gs_torch(
        targets_subset, depths_subset, WAVELENGTH, DX, N_ITERS_BUDGET,
        device=DEVICE, seed=seed, pad_factor=PAD_FACTOR, smooth_cutoff=True
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    plateau_iter, reference, still_rising = find_plateau_robust(history)

    final_checkpoint = max(recon_by_checkpoint.keys())
    recon_planes = recon_by_checkpoint[final_checkpoint]
    corrected_psnrs = []
    for r, t, global_idx in zip(recon_planes, targets_subset, plane_idxs):
        if global_idx in SPARSE_PLANE_IDXS:
            rows, cols = sparse_content_bounds(t)
            corrected_psnrs.append(psnr_intensity(r[rows, cols], t[rows, cols]))
        else:
            corrected_psnrs.append(psnr_intensity(r, t))
    corrected_avg = sum(corrected_psnrs) / len(corrected_psnrs)

    return {"plateau_iter": plateau_iter, "still_rising": still_rising,
            "corrected_avg": corrected_avg, "per_plane": corrected_psnrs, "elapsed": elapsed}


def main():
    print(f"Device: {DEVICE}")
    t_start = time.time()
    targets = make_realistic_multiplane_target(SHAPE)

    print("Warming up GPU...")
    _ = multiplane_gs_torch(targets, DEPTHS_M, WAVELENGTH, DX, 1, device=DEVICE,
                             pad_factor=PAD_FACTOR, smooth_cutoff=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    all_results = {}
    for cond_name, plane_idxs in CONDITIONS.items():
        print(f"\n{cond_name}...")
        targets_subset = [targets[i] for i in plane_idxs]
        depths_subset = [DEPTHS_M[i] for i in plane_idxs]
        runs = []
        for seed in SEEDS:
            r = solve(targets_subset, depths_subset, plane_idxs, seed)
            runs.append(r)
            per_plane_str = ", ".join(f"{v:.2f}" for v in r["per_plane"])
            print(f"   seed={seed}: corrected={r['corrected_avg']:.2f} dB (per-plane: {per_plane_str}), "
                  f"plateau={r['plateau_iter']}, {r['elapsed']:.1f}s"
                  f"{' STILL RISING -- N_ITERS may be too low' if r['still_rising'] else ''}")
        all_results[cond_name] = runs

    print("\n" + "=" * 90)
    print(f"{'Condition':<28}{'Seeds (dB)':<32}{'Avg':<10}{'Range'}")
    for cond_name, runs in all_results.items():
        vals = [r["corrected_avg"] for r in runs]
        avg = sum(vals) / len(vals)
        print(f"{cond_name:<28}{str([round(v,2) for v in vals]):<32}{avg:<10.2f}"
              f"{min(vals):.2f}-{max(vals):.2f}")
    print("=" * 90)

    any_rising = any(r["still_rising"] for runs in all_results.values() for r in runs)
    if any_rising:
        print("\nWARNING: at least one run was still rising at N_ITERS_BUDGET -- consider a")
        print("longer budget before fully trusting the averages above.")

    avgs = {name: sum(r["corrected_avg"] for r in runs) / len(runs) for name, runs in all_results.items()}
    names = list(CONDITIONS.keys())
    print(f"\nOriginal single-seed claim: 14.6 dB (1 plane) -> 11.9 dB (2 planes) -> 11.5 dB (3 planes)")
    print(f"3-seed average, full resolution: {avgs[names[0]]:.2f} dB -> {avgs[names[1]]:.2f} dB "
          f"-> {avgs[names[2]]:.2f} dB")

    monotonic = avgs[names[0]] > avgs[names[1]] > avgs[names[2]]
    if monotonic:
        print("\nMonotonic degradation confirmed at full resolution, multiple seeds -- the original")
        print("finding holds under the corrected methodology.")
    else:
        print("\nNOT monotonic under the corrected methodology -- the original ordering does not")
        print("hold up; report the actual ordering/ranges above instead.")

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min")

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (cond_name, runs) in enumerate(all_results.items()):
        vals = [r["corrected_avg"] for r in runs]
        ax.scatter([i] * len(vals), vals, alpha=0.7)
        ax.scatter([i], [sum(vals) / len(vals)], color="red", marker="_", s=400, linewidths=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["1 plane", "2 planes", "3 planes"])
    ax.set_ylabel("Corrected PSNR, avg across solved planes (dB)")
    ax.set_title(f"Plane-count quality degradation, {SHAPE[1]}x{SHAPE[0]}, {len(SEEDS)} seeds")
    plt.tight_layout()
    plt.savefig("plane_count_seed_check.png", dpi=130)
    print("Saved plot: plane_count_seed_check.png")


if __name__ == "__main__":
    main()
