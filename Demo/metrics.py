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


def psnr_intensity(recon_amp, target_intensity, eps=1e-8):
    """
    Reconstructions from multiplane_gs_torch/multiplane_sgd are field
    AMPLITUDE (safe_abs of the propagated field), but the target arrays
    from targets.py (make_multiplane_target, make_realistic_multiplane_target)
    are INTENSITY (0..1 images) -- the training loop internally compares
    amplitude to amplitude (targets_amp = sqrt(target_intensity), see
    _to_tensor_targets in retrieval_torch.py), but psnr() above was being
    called directly against the raw intensity target, an amplitude-vs-
    intensity units mismatch that silently distorts the reported dB
    (confirmed by comparing training-loop history against externally
    recomputed PSNR on the same reconstruction: they disagreed by up to
    ~2.7dB). Square the amplitude reconstruction to intensity first so
    both sides of the comparison are the same physical quantity.
    """
    return psnr(recon_amp ** 2, target_intensity, eps=eps)
