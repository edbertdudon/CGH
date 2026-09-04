"""
validate_propagation.py

Sanity check for AngularSpectrumPropagation, matching milestone 2 from
the roadmap: propagate a phase-only lens pattern and confirm it focuses
at the expected distance z = f, the classic thin-lens focusing test.

This also mirrors the phase-wrapping discussion directly: the lens phase
is built as the smooth, unbounded "ideal" ramp, then wrapped into [0, 2pi)
before use, since that's exactly what a real phase-only SLM would show.

Run this on the RTX 3060 (needs a real CUDA install of torch; this
sandbox has no network access to install torch, so the underlying math
was validated separately with numpy -- see the conversation).
"""

import torch
from angular_spectrum_propagation import AngularSpectrumPropagation

# --- physical parameters ---
wavelength = 532e-9    # green light, meters
pixel_pitch = 8e-6     # 8 micrometers -- matches a typical LCOS SLM pitch
resolution = (512, 512)
focal_length = 0.05    # 5 cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

propagator = AngularSpectrumPropagation(
    resolution=resolution,
    pixel_pitch=pixel_pitch,
    wavelength=wavelength,
    device=device,
)

# --- build a thin-lens phase pattern ---
ny, nx = resolution
y = (torch.arange(ny, device=device) - ny // 2) * pixel_pitch
x = (torch.arange(nx, device=device) - nx // 2) * pixel_pitch
Y, X = torch.meshgrid(y, x, indexing="ij")

# Ideal continuous lens phase -- the "smooth ramp" from the phase-wrapping
# discussion. Wrapping it into [0, 2*pi) is physically exact, not a loss.
lens_phase = -(torch.pi / (wavelength * focal_length)) * (X**2 + Y**2)
wrapped_phase = torch.remainder(lens_phase, 2 * torch.pi)

# Phase-only field: amplitude 1 everywhere, phase = wrapped lens phase.
field0 = torch.exp(1j * wrapped_phase)

# --- propagate across a range of distances and find where it focuses ---
z_values = torch.linspace(0.3 * focal_length, 1.7 * focal_length, 60)
peak_intensities = []

with torch.no_grad():
    for z in z_values:
        Uz = propagator(field0, z.item())
        intensity = Uz.abs() ** 2
        peak_intensities.append(intensity.max().item())

peak_intensities = torch.tensor(peak_intensities)
best_z = z_values[torch.argmax(peak_intensities)].item()

print(f"Expected focal distance (f):   {focal_length * 1000:.2f} mm")
print(f"Distance of peak intensity:    {best_z * 1000:.2f} mm")
print(f"Relative error:                {abs(best_z - focal_length) / focal_length * 100:.2f}%")

# --- energy conservation check ---
e0 = (field0.abs() ** 2).sum().item()
Uz_f = propagator(field0, focal_length)
ef = (Uz_f.abs() ** 2).sum().item()
print(f"Energy at z=0:  {e0:.4f}")
print(f"Energy at z=f:  {ef:.4f}")
print(f"Energy ratio:   {ef / e0:.4f}  (should be ~1.0 -- propagation is lossless)")
