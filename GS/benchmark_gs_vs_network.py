"""
benchmark_gs_vs_network.py

The comparison the roadmap has been building toward: the same target
image, the same physics, two different ways of finding a phase pattern --
200 rounds of Gerchberg-Saxton versus one forward pass of the trained
network. Compares both how long it takes to find the pattern and how
good the resulting reconstruction is.
"""

import time
import torch
from angular_spectrum_propagation import AngularSpectrumPropagation
from gerchberg_saxton import gerchberg_saxton
from phase_net import PhaseNet
from train_phase_net import make_random_shape_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

wavelength = 532e-9
pixel_pitch = 8e-6
resolution = (256, 256)
z = 0.05

propagator = AngularSpectrumPropagation(
    resolution=resolution, pixel_pitch=pixel_pitch, wavelength=wavelength, device=device
)

model = PhaseNet().to(device)
model.load_state_dict(torch.load("phase_net_weights.pt", map_location=device))
model.eval()


def cosine_similarity(a, b):
    a_flat, b_flat = a.flatten(), b.flatten()
    a_norm = a_flat / torch.linalg.vector_norm(a_flat)
    b_norm = b_flat / torch.linalg.vector_norm(b_flat)
    return torch.dot(a_norm, b_norm).item()


# same, unseen target for both methods
generator = torch.Generator(device="cpu").manual_seed(999)
target_batch = make_random_shape_batch(1, resolution, device, generator)
target = target_batch[0, 0]

# --- Gerchberg-Saxton: finding the pattern takes 200 rounds of propagation ---
t0 = time.time()
gs_phase, gs_errors = gerchberg_saxton(target, propagator, z, n_iters=200)
gs_find_time = time.time() - t0
with torch.no_grad():
    gs_recon = propagator(torch.exp(1j * gs_phase), z).abs()
gs_similarity = cosine_similarity(gs_recon, target)

# --- Network: finding the pattern is one forward pass ---
with torch.no_grad():
    t0 = time.time()
    raw_phase = model(target_batch).squeeze(1)
    net_find_time = time.time() - t0
    net_phase = torch.remainder(raw_phase, 2 * torch.pi)[0]
    net_recon = propagator(torch.exp(1j * net_phase), z).abs()
net_similarity = cosine_similarity(net_recon, target)

print(f"Gerchberg-Saxton (200 iters): {gs_find_time * 1000:8.1f} ms, similarity {gs_similarity:.3f}")
print(f"Network (1 forward pass):    {net_find_time * 1000:8.1f} ms, similarity {net_similarity:.3f}")
print(f"Speedup: {gs_find_time / net_find_time:.0f}x faster")

# --- save a visual comparison ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(target.cpu().numpy(), cmap="gray")
axes[0].set_title("Target")
axes[1].imshow(gs_recon.cpu().numpy(), cmap="gray")
axes[1].set_title(f"Gerchberg-Saxton\n{gs_find_time * 1000:.0f} ms, sim {gs_similarity:.2f}")
axes[2].imshow(net_recon.cpu().numpy(), cmap="gray")
axes[2].set_title(f"Network\n{net_find_time * 1000:.0f} ms, sim {net_similarity:.2f}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("benchmark_comparison.png", dpi=150)
print("Saved benchmark_comparison.png")
