"""
quantization_test.py

Tests how much reconstruction quality is lost when a real SLM's limited
number of phase levels replaces the continuous, full-precision phase
values everything so far has assumed. Every phase value in this whole
project has been a full float -- infinite resolution, which no real
hardware has.

Real phase-only SLMs only offer a fixed number of discrete phase steps,
set by the bit-depth of their driving electronics: an 8-bit SLM has 256
levels, a cheaper 4-bit one has 16, a bare-minimum 2-bit one has 4.
Quantizing rounds each pixel's phase down to the nearest of those levels
-- unlike phase wrapping (which is exact), this rounding step genuinely
throws information away.
"""

import torch
from angular_spectrum_propagation import AngularSpectrumPropagation
from gerchberg_saxton import gerchberg_saxton

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

wavelength = 532e-9
pixel_pitch = 8e-6
resolution = (256, 256)
z = 0.05

propagator = AngularSpectrumPropagation(
    resolution=resolution, pixel_pitch=pixel_pitch, wavelength=wavelength, device=device
)


def cosine_similarity(a, b):
    a_flat, b_flat = a.flatten(), b.flatten()
    a_norm = a_flat / torch.linalg.vector_norm(a_flat)
    b_norm = b_flat / torch.linalg.vector_norm(b_flat)
    return torch.dot(a_norm, b_norm).item()


def quantize_phase(phase, n_bits):
    """Round continuous phase down to the nearest of 2**n_bits discrete
    levels spanning [0, 2*pi) -- exactly what a real SLM's driving
    electronics do."""
    n_levels = 2 ** n_bits
    step = 2 * torch.pi / n_levels
    return torch.remainder(torch.round(phase / step) * step, 2 * torch.pi)


# --- the same smiley-face target used in the Gerchberg-Saxton test ---
ny, nx = resolution
yy, xx = torch.meshgrid(
    torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
)
cy, cx = ny // 2, nx // 2
target = torch.zeros(ny, nx, device=device)
r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
target[(r > 80) & (r < 90)] = 1.0
target[torch.sqrt((xx - cx + 30) ** 2 + (yy - cy + 30) ** 2) < 8] = 1.0
target[torch.sqrt((xx - cx - 30) ** 2 + (yy - cy + 30) ** 2) < 8] = 1.0
mouth_r = torch.sqrt((xx - cx) ** 2 + (yy - cy - 10) ** 2)
target[(mouth_r > 40) & (mouth_r < 48) & (yy > cy + 10)] = 1.0

# --- solve for a full-precision phase pattern first ---
full_phase, _ = gerchberg_saxton(target, propagator, z, n_iters=200)
with torch.no_grad():
    full_recon = propagator(torch.exp(1j * full_phase), z).abs()
full_similarity = cosine_similarity(full_recon, target)

# --- quantize to realistic bit depths and re-check quality ---
bit_depths = [8, 6, 4, 3, 2, 1]
results, recons = {}, {}

for bits in bit_depths:
    q_phase = quantize_phase(full_phase, bits)
    with torch.no_grad():
        q_recon = propagator(torch.exp(1j * q_phase), z).abs()
    sim = cosine_similarity(q_recon, target)
    results[bits] = sim
    recons[bits] = q_recon.cpu().numpy()
    print(f"{bits}-bit ({2 ** bits:4d} levels): similarity {sim:.4f}  (full precision: {full_similarity:.4f})")

# --- visual comparison across bit depths ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(bit_depths) + 1, figsize=(3 * (len(bit_depths) + 1), 3))
axes[0].imshow(full_recon.cpu().numpy(), cmap="gray")
axes[0].set_title(f"Full precision\nsim {full_similarity:.3f}")
for i, bits in enumerate(bit_depths):
    axes[i + 1].imshow(recons[bits], cmap="gray")
    axes[i + 1].set_title(f"{bits}-bit\nsim {results[bits]:.3f}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("quantization_test.png", dpi=150)
print("Saved quantization_test.png")

# --- quality vs bit depth curve ---
fig2, ax2 = plt.subplots(figsize=(6, 4))
xs = bit_depths[::-1]
ys = [results[b] for b in xs]
ax2.plot(xs, ys, marker="o")
ax2.axhline(full_similarity, color="gray", linestyle="--", label="full precision")
ax2.set_xlabel("SLM bit depth")
ax2.set_ylabel("Reconstruction similarity")
ax2.set_title("Quality vs. phase quantization")
ax2.legend()
plt.tight_layout()
plt.savefig("quantization_curve.png", dpi=150)
print("Saved quantization_curve.png")
