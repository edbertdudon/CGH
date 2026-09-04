"""
angular_spectrum_propagation.py

A differentiable implementation of the Angular Spectrum Method (ASM) for
scalar wave propagation -- the same FFT -> multiply by H -> IFFT pipeline
we derived by hand:

    U(x, y, z) = IFFT{ FFT{U(x, y, 0)} * H(fx, fy; z) }

where H comes directly from the fx^2 + fy^2 + fz^2 = 1/lambda^2 constraint:

    H(fx, fy; z) = exp(i * 2*pi * z * sqrt(1/lambda^2 - fx^2 - fy^2))

Frequencies with fx^2 + fy^2 >= 1/lambda^2 are evanescent (fz would be
imaginary) and are zeroed out, matching the discussion of why those
components decay instead of propagate.

Built as an nn.Module so it drops straight into a training loop: gradients
flow through the FFT, the multiply, and the inverse FFT, exactly like the
"ASM layer" box in the training-loop diagram from earlier.
"""

import torch
import torch.nn as nn


class AngularSpectrumPropagation(nn.Module):
    """
    Differentiable free-space propagation of a complex scalar field.

    Parameters
    ----------
    resolution : tuple[int, int]
        (height, width) of the field, in pixels.
    pixel_pitch : float
        Physical size of one pixel (meters). Use the same units as
        wavelength throughout -- e.g. both in meters.
    wavelength : float
        Wavelength of light (meters).
    device : torch.device, optional
        Device to precompute the frequency grid on.
    """

    def __init__(self, resolution, pixel_pitch, wavelength, device=None):
        super().__init__()
        self.ny, self.nx = resolution
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        device = device or torch.device("cpu")

        # Spatial frequency grid (fx, fy) -- one entry per pixel, in
        # cycles per unit length. This is the "which stripe pattern"
        # decomposition FFT will produce.
        fy = torch.fft.fftfreq(self.ny, d=pixel_pitch)
        fx = torch.fft.fftfreq(self.nx, d=pixel_pitch)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")

        # fz^2 = 1/lambda^2 - fx^2 - fy^2, straight from the triangle:
        # fx^2 + fy^2 + fz^2 = 1/lambda^2.
        fz_squared = (1.0 / wavelength) ** 2 - FX**2 - FY**2

        # Propagating band: fz_squared >= 0. Everything past that is the
        # evanescent band -- fz would be imaginary, so we zero it out
        # rather than let it produce a spurious growing/decaying term.
        propagating_mask = fz_squared >= 0
        fz = torch.zeros_like(fz_squared)
        fz[propagating_mask] = torch.sqrt(fz_squared[propagating_mask])

        # Stored once; H itself is rebuilt per z at call time, since z
        # varies (e.g. different depth planes for a 3D scene).
        self.register_buffer("fz", fz.to(device))
        self.register_buffer("propagating_mask", propagating_mask.to(device))

    def transfer_function(self, z):
        """Build H(fx, fy; z): spin every frequency component's arrow by
        its own required amount for having traveled distance z."""
        phase = 2 * torch.pi * self.fz * z
        H = torch.exp(1j * phase)
        H = H * self.propagating_mask
        return H

    def forward(self, field, z):
        """
        Propagate a complex field by distance z.

        Parameters
        ----------
        field : torch.Tensor, complex dtype, shape (..., ny, nx)
            The input field, U(x, y, 0).
        z : float
            Propagation distance, same units as wavelength/pixel_pitch.

        Returns
        -------
        torch.Tensor, complex dtype, shape (..., ny, nx)
            The propagated field, U(x, y, z).
        """
        H = self.transfer_function(z)
        A = torch.fft.fft2(field)        # decompose into stripe components
        A_prop = A * H                   # spin each component's arrow
        U_prop = torch.fft.ifft2(A_prop) # add the spun arrows back together
        return U_prop
