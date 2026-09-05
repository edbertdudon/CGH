"""
Free-space wave propagation.

Scope note: this models scalar diffraction only (angular spectrum method).
It does NOT model the real waveguide -- no phase distortion across pupil
replication, no coupling loss, no per-unit calibration error. Those numbers
don't exist yet (they're Gap 4 in the technical reference doc, owned by the
waveguide house). This module answers "how many FFTs / iterations does the
phase-retrieval algorithm itself need," which is independent of whether the
waveguide problem is ever solved.
"""
import numpy as np
from fft_counter import counter


def angular_spectrum_propagate(field, wavelength, dx, z, pad_factor=2):
    """
    Propagate a complex field by distance z using the angular spectrum
    method. Zero-pads before propagating and crops back after, to reduce
    FFT wraparound artifacts when z is large relative to the aperture
    (which it easily can be here -- a 2um-pitch SLM has a native diffraction
    half-angle of several degrees, so the beam spreads fast).

    field: 2D complex array, the field at the source plane
    wavelength: meters
    dx: sample pitch, meters (square pixels assumed)
    z: propagation distance, meters (negative = propagate backward)
    """
    ny, nx = field.shape
    py, px = ny * pad_factor, nx * pad_factor
    padded = np.zeros((py, px), dtype=complex)
    oy, ox = (py - ny) // 2, (px - nx) // 2
    padded[oy:oy + ny, ox:ox + nx] = field

    fx = np.fft.fftfreq(px, d=dx)
    fy = np.fft.fftfreq(py, d=dx)
    FX, FY = np.meshgrid(fx, fy)

    arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    propagating = arg >= 0
    kz = np.zeros_like(arg)
    kz[propagating] = 2 * np.pi * np.sqrt(arg[propagating]) / wavelength
    H = np.zeros_like(arg, dtype=complex)
    H[propagating] = np.exp(1j * kz[propagating] * z)

    F = counter.fft2(padded)
    F_prop = F * H
    out_padded = counter.ifft2(F_prop)

    return out_padded[oy:oy + ny, ox:ox + nx]


def lens_fourier_transform(field):
    """
    Idealized final Fourier transform performed by the eye's lens in the
    pupil-plane (Config B) architecture: SLM is conjugated to the pupil,
    the eye lens performs the last Fourier transform onto the retina.

    Simplification: treated as a single ideal 2D FFT (no lens aberration,
    no pupil apodization). Real retinal image quality also depends on
    natural eye aberrations and pupil size, which this does not model.
    """
    return np.fft.fftshift(counter.fft2(np.fft.ifftshift(field)))
