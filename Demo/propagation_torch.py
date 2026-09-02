"""
Torch port of propagation.py. Deliberately mirrors the numpy version
line-for-line where possible -- the numpy version was validated first
(see propagation.py / demo.py results), and this should behave the same
up to floating-point precision differences (numpy ran float64, this runs
complex64 for GPU speed/memory).

Runs on CPU too (device='cpu') as a smoke test if you don't have a GPU
handy, just slower -- useful for checking it imports and runs correctly
before trusting a full GPU run.
"""
import math
import torch
from fft_counter_torch import counter


def angular_spectrum_propagate(field, wavelength, dx, z, pad_factor=2, complex_dtype=torch.complex64,
                                smooth_cutoff=False, cutoff_width=0.05):
    """
    field: 2D complex torch tensor (dtype should match complex_dtype), on
    whatever device you want to run on
    wavelength, dx, z: python floats (meters)
    complex_dtype: torch.complex64 (default, fast) or torch.complex128
    (double precision -- much slower on consumer GPUs, use only to test
    whether an artifact is precision-related).
    smooth_cutoff: if True, replace the hard propagating/evanescent
    boolean mask with a smooth ramp of width cutoff_width (in the same
    units as `arg` below, i.e. 1 - (lambda*fx)^2 - (lambda*fy)^2). The
    hard mask is a sharp discontinuity in frequency space; at high
    resolution, frequency sampling is fine enough to resolve that edge
    clearly, which can produce visible ringing after the inverse FFT.
    Test this against smooth_cutoff=False at matched resolution to check.
    """
    device = field.device
    ny, nx = field.shape
    py, px = int(ny * pad_factor), int(nx * pad_factor)
    padded = torch.zeros((py, px), dtype=complex_dtype, device=device)
    oy, ox = (py - ny) // 2, (px - nx) // 2
    padded[oy:oy + ny, ox:ox + nx] = field

    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    fx = torch.fft.fftfreq(px, d=dx, device=device).to(real_dtype)
    fy = torch.fft.fftfreq(py, d=dx, device=device).to(real_dtype)
    # indexing='ij' + (fy, fx) order reproduces numpy.meshgrid(fx, fy)
    # (numpy's default 'xy' indexing) -- see propagation.py for the
    # numpy version this mirrors.
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    safe_arg = torch.clamp(arg, min=0.0)
    kz = 2 * math.pi * torch.sqrt(safe_arg) / wavelength

    if smooth_cutoff:
        # smooth ramp from 0 (deep evanescent) to 1 (propagating), instead
        # of a hard boolean edge -- avoids a sharp frequency-domain
        # discontinuity
        edge = torch.clamp(arg / cutoff_width, 0.0, 1.0)
        H = edge.to(complex_dtype) * torch.exp(1j * kz.to(real_dtype) * z)
    else:
        propagating = arg >= 0
        kz_masked = torch.zeros_like(arg)
        kz_masked[propagating] = kz[propagating]
        H = torch.zeros_like(arg, dtype=complex_dtype)
        H[propagating] = torch.exp(1j * kz_masked[propagating] * z)

    F = counter.fft2(padded)
    F_prop = F * H
    out_padded = counter.ifft2(F_prop)

    return out_padded[oy:oy + ny, ox:ox + nx]


def safe_abs(field, eps=1e-12):
    """
    torch's autograd through .abs() on complex tensors can produce NaN
    gradients exactly where magnitude is 0 (the derivative of sqrt(x) at
    x=0 is undefined). Use this instead of field.abs() anywhere gradients
    need to flow through it (i.e. inside the SGD loop). Safe to use
    everywhere else too -- it's numerically identical except at 0.
    """
    return torch.sqrt(field.real ** 2 + field.imag ** 2 + eps)
