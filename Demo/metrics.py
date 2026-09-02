"""
PSNR metric, numpy version -- mirrors _psnr_torch in retrieval_torch.py
(each image normalized by its own max before comparing) so results are
directly comparable across the torch and numpy code paths.
"""
import numpy as np


def psnr(recon_amp, target_amp, eps=1e-8):
    r = recon_amp / (recon_amp.max() + eps)
    t = target_amp / (target_amp.max() + eps)
    mse = np.mean((r - t) ** 2)
    if mse < eps:
        return 99.0
    return float(10 * np.log10(1.0 / mse))
