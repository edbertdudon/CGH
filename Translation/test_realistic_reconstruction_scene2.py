"""
First CGH reconstruction test on real, non-procedural content: the
depth-bucketed real photo + Japanese-caption planes built by
build_realistic_target.py, run through the same Adam/SGD phase-retrieval
solver used throughout this project.

Open question this answers: every prior quality/compute number in this
project was measured on procedurally-generated content (a clean icon, a
short rendered caption, a smooth synthetic gradient scene) -- flagged
more than once as untested against real, messier, natural-image content
with genuine texture and fine detail. This is the first real answer.

All 3 planes here are naturally sparse (each is a masked cutout of the
same photo, mostly black elsewhere) unlike the old procedural convention
where "far" filled the whole frame -- so sparse_content_bounds masking is
applied to all 3 planes, not just near/mid, to avoid the same "trivially-
correct background inflates whole-frame PSNR" bug this project has hit
and fixed twice before.

Generous fixed budget + plateau detection (not a fixed guessed budget),
per this project's standing rule not to trust a budget without checking
both conditions actually converged.

Run on your 3060 (expect ~1 min for 5 seeds):
    python3 test_realistic_reconstruction.py
"""
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "../Demo")
from retrieval_torch import _to_tensor_targets
from propagation_torch import angular_spectrum_propagate, safe_abs
from metrics import psnr_intensity
from convergence import find_plateau_robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]  # near/mid/far proxy depths, same convention as every other test
N_STEPS_MAX = 800
LR = 0.02
SEEDS = [0, 1, 2, 3, 4]


def load_targets():
    near = np.load("target_near_2289.npy")
    mid = np.load("target_mid_2289.npy")
    far = np.load("target_far_2289.npy")
    return [near, mid, far]


def masked_quality(recon_planes_np, targets_np, content_thresh=0.05):
    """
    Per-pixel content mask (target > content_thresh), NOT a rectangular
    bounding box: this project's real, depth-bucketed photo content is
    scattered across the frame in disconnected regions (background
    pillars, ceiling structure, etc.), unlike the old procedural
    convention's single compact icon/caption blob that a bounding box
    could tightly crop. A bounding box around scattered content still
    contains mostly empty background (confirmed directly: the "mid"
    plane's box covered 83.6% of the frame but was only 16.5% nonzero,
    "far" covered 97.1% i.e. almost the whole frame) -- the same
    background-domination bug this project has hit before (A.6, 10.8),
    just in a new shape. Masking every actual content pixel individually
    fixes it regardless of how the content is distributed spatially.
    """
    psnrs = []
    for r, t in zip(recon_planes_np, targets_np):
        mask = t > content_thresh
        psnrs.append(psnr_intensity(r[mask], t[mask]))
    return psnrs


def run_seed(targets_np, seed):
    shape = targets_np[0].shape
    targets_amp = _to_tensor_targets(targets_np, DEVICE)
    torch.manual_seed(seed)
    slm_phase = (torch.rand(shape, device=DEVICE) * 2 * np.pi - np.pi).requires_grad_(True)
    opt = torch.optim.Adam([slm_phase], lr=LR)

    history = []
    for step in range(N_STEPS_MAX):
        opt.zero_grad()
        field = torch.exp(1j * slm_phase)
        loss = torch.tensor(0.0, device=DEVICE)
        recon_planes = []
        for target_amp, z in zip(targets_amp, DEPTHS_M):
            recon = safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR))
            loss = loss + torch.mean((recon - target_amp) ** 2)
            recon_planes.append(recon.detach().cpu().numpy())
        loss.backward()
        opt.step()

        q = masked_quality(recon_planes, targets_np)
        history.append(sum(q) / len(q))

    plateau_iter, reference_quality, still_rising = find_plateau_robust(history)
    return {
        "history": history,
        "plateau_iter": plateau_iter,
        "reference_quality": reference_quality,
        "still_rising": still_rising,
        "final_recon": recon_planes,
    }


def main():
    print(f"Device: {DEVICE}")
    targets_np = load_targets()
    print(f"Target shapes: {[t.shape for t in targets_np]}")

    t_start = time.time()
    results = []
    for seed in SEEDS:
        t0 = time.time()
        r = run_seed(targets_np, seed)
        results.append(r)
        flag = " [STILL RISING -- budget may be too short]" if r["still_rising"] else ""
        print(f"   seed {seed}: converged quality={r['reference_quality']:.2f} dB, "
              f"plateau at iter {r['plateau_iter']}/{N_STEPS_MAX}{flag} [{time.time()-t0:.1f}s]")

    qualities = [r["reference_quality"] for r in results]
    print(f"\n{'='*80}")
    print(f"Real-content reconstruction: mean={np.mean(qualities):.2f} dB (std={np.std(qualities):.2f}), "
          f"range={min(qualities):.2f}-{max(qualities):.2f} dB")
    n_still_rising = sum(1 for r in results if r["still_rising"])
    print(f"Seeds still rising at budget end: {n_still_rising}/{len(SEEDS)} "
          f"{'(trust these numbers)' if n_still_rising == 0 else '(budget may need extending)'}")
    print(f"Total runtime: {time.time()-t_start:.1f}s")

    # save a reconstruction preview from the last seed for visual inspection
    from PIL import Image
    last = results[-1]["final_recon"]
    names = ["near", "mid", "far"]
    for name, arr in zip(names, last):
        Image.fromarray((np.clip(arr / (arr.max() + 1e-8), 0, 1) * 255).astype(np.uint8)).save(f"recon_{name}.png")
    print("Saved reconstruction previews: recon_near.png, recon_mid.png, recon_far.png")


if __name__ == "__main__":
    main()
