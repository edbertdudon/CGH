"""
Diagnostic: the main experiment (experiment_subaperture_vs_discrete_tile.py)
found "tile_flicker" (averaging all 4 discrete-tile patterns' reconstructed
amplitudes) scoring HIGHER (8.59 dB) than either a viewpoint-matched single
tile (7.55 dB) or the jointly-trained pattern (7.39 dB) -- at every test
viewpoint, including ones far from 3 of the 4 tiles' own training centers.
That's suspicious: averaging in badly-mismatched content should hurt, not
help, unless the boost is actually just noise-averaging (this project's
10.2 already found averaging independent phase-retrieval solutions reduces
speckle a little, regardless of what they're solutions TO).

This checks that directly: train 4 patterns, different random seeds, all
against the SAME single fixed viewpoint (no parallax shift at all -- so
there's no "viewpoint matching" question, only "does averaging 4 independent
noisy solutions to the identical problem boost dB"). If this ALSO shows a
similar boost, the flicker-realistic result above is a generic denoising
artifact, not evidence about viewpoint generalization, and needs to be
reported as such rather than taken at face value.

Run on your 3060 (expect ~1 min):
    python3 experiment_flicker_diagnostic.py
"""
import numpy as np
import torch

from targets import make_realistic_multiplane_target, sparse_content_bounds
from retrieval_torch import _to_tensor_targets
from propagation_torch import angular_spectrum_propagate, safe_abs
from metrics import psnr_intensity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVELENGTH = 520e-9
DX = 2.0e-6
SHAPE = (512, 512)
PAD_FACTOR = 2.0
DEPTHS_M = [1.0e-3, 3.0e-3, 6.0e-3]
N_STEPS = 500
LR = 0.02
N_PATTERNS = 4


def masked_quality(recon_planes_np, targets_np):
    psnrs = []
    for i, (r, t) in enumerate(zip(recon_planes_np, targets_np)):
        if i in (0, 1):
            rows, cols = sparse_content_bounds(t)
            psnrs.append(psnr_intensity(r[rows, cols], t[rows, cols]))
        else:
            psnrs.append(psnr_intensity(r, t))
    return psnrs


def train_fixed(targets_np, seed):
    targets_amp = _to_tensor_targets(targets_np, DEVICE)
    torch.manual_seed(seed)
    slm_phase = (torch.rand(SHAPE, device=DEVICE) * 2 * np.pi - np.pi).requires_grad_(True)
    opt = torch.optim.Adam([slm_phase], lr=LR)
    for step in range(N_STEPS):
        opt.zero_grad()
        field = torch.exp(1j * slm_phase)
        loss = torch.tensor(0.0, device=DEVICE)
        for target_amp, z in zip(targets_amp, DEPTHS_M):
            recon = safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR))
            loss = loss + torch.mean((recon - target_amp) ** 2)
        loss.backward()
        opt.step()
    return slm_phase.detach()


def propagate_all(slm_phase):
    field = torch.exp(1j * slm_phase)
    return [safe_abs(angular_spectrum_propagate(field, WAVELENGTH, DX, z, pad_factor=PAD_FACTOR)) for z in DEPTHS_M]


def main():
    print(f"Device: {DEVICE}")
    base_planes = make_realistic_multiplane_target(SHAPE)

    print(f"Training {N_PATTERNS} independent patterns, different seeds, SAME single fixed viewpoint (no shift)...")
    phases = []
    for i in range(N_PATTERNS):
        p = train_fixed(base_planes, seed=100 + i)
        phases.append(p)
        recon = [r.detach().cpu().numpy() for r in propagate_all(p)]
        q = masked_quality(recon, base_planes)
        print(f"   pattern {i}: individual quality = {sum(q)/len(q):.2f} dB")

    all_recons = [propagate_all(p) for p in phases]
    avg_recon = [
        torch.stack([all_recons[t][plane_idx] for t in range(N_PATTERNS)]).mean(dim=0).cpu().numpy()
        for plane_idx in range(3)
    ]
    q_avg = masked_quality(avg_recon, base_planes)
    individual_avgs = []
    for i in range(N_PATTERNS):
        recon = [r.detach().cpu().numpy() for r in all_recons[i]]
        q = masked_quality(recon, base_planes)
        individual_avgs.append(sum(q) / len(q))

    print(f"\nIndividual pattern quality: {[round(v,2) for v in individual_avgs]}, "
          f"mean={sum(individual_avgs)/len(individual_avgs):.2f} dB")
    print(f"Averaged (4-pattern flicker) quality: {sum(q_avg)/len(q_avg):.2f} dB")

    boost = sum(q_avg)/len(q_avg) - sum(individual_avgs)/len(individual_avgs)
    print(f"\nBoost from averaging 4 independent same-viewpoint solutions: {boost:+.2f} dB")
    if boost > 0.5:
        print("\nCONFIRMED: averaging independent phase-retrieval solutions boosts dB even with ZERO")
        print("viewpoint difference between them -- this is generic speckle-averaging (matches 10.2's")
        print("earlier finding), not evidence of viewpoint generalization. The main experiment's")
        print("tile_flicker result is contaminated by this effect and should not be read as showing")
        print("the discrete-tile scheme genuinely handles off-center viewpoints well.")
    else:
        print("\nNo meaningful boost from averaging alone -- the main experiment's tile_flicker result")
        print("is not just a denoising artifact; worth taking more seriously as a real effect.")


if __name__ == "__main__":
    main()
