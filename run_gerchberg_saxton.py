"""
run_gerchberg_saxton.py

Builds a simple target image (a smiley face -- easy to eyeball whether the
reconstruction actually worked) and runs Gerchberg-Saxton against the
already-validated AngularSpectrumPropagation layer to find a phase-only
SLM pattern that reproduces it. Saves a side-by-side image so you can
actually look at the result.
"""

import torch
from angular_spectrum_propagation import AngularSpectrumPropagation
from gerchberg_saxton import gerchberg_saxton

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# --- physical parameters (same as the propagation validation) ---
wavelength = 532e-9
pixel_pitch = 8e-6
resolution = (256, 256)
z = 0.05

propagator = AngularSpectrumPropagation(
    resolution=resolution,
    pixel_pitch=pixel_pitch,
    wavelength=wavelength,
    device=device,
)

# --- build a simple smiley-face target amplitude ---
ny, nx = resolution
yy, xx = torch.meshgrid(
    torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
)
cy, cx = ny // 2, nx // 2
target = torch.zeros(ny, nx, device=device)

r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
target[(r > 80) & (r < 90)] = 1.0  # face outline ring
target[torch.sqrt((xx - cx + 30) ** 2 + (yy - cy + 30) ** 2) < 8] = 1.0  # left eye
target[torch.sqrt((xx - cx - 30) ** 2 + (yy - cy + 30) ** 2) < 8] = 1.0  # right eye
mouth_r = torch.sqrt((xx - cx) ** 2 + (yy - cy - 10) ** 2)
target[(mouth_r > 40) & (mouth_r < 48) & (yy > cy + 10)] = 1.0  # smile arc

# --- run Gerchberg-Saxton ---
slm_phase, errors = gerchberg_saxton(target, propagator, z, n_iters=200)

print(f"Initial reconstruction error: {errors[0]:.4f}")
print(f"Error after 10 iterations:    {errors[10]:.4f}")
print(f"Error after 50 iterations:    {errors[50]:.4f}")
print(f"Final error (200 iters):      {errors[-1]:.4f}")

# --- save a side-by-side comparison so you can actually look at it ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with torch.no_grad():
    final_field = propagator(torch.exp(1j * slm_phase), z)
    final_amp = final_field.abs().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(target.cpu().numpy(), cmap="gray")
axes[0].set_title("Target")
axes[1].imshow(slm_phase.cpu().numpy() % (2 * torch.pi), cmap="twilight")
axes[1].set_title("Recovered SLM phase")
axes[2].imshow(final_amp, cmap="gray")
axes[2].set_title("Reconstruction")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("gerchberg_saxton_result.png", dpi=150)
print("Saved gerchberg_saxton_result.png")
