"""
eval_phase_net_depth.py

The actual test of depth: propagate the SAME predicted phase pattern to
two different distances and check that each one shows the right content
in focus. This is what makes it a 3D hologram rather than a picture
glued to a fixed distance -- one phase pattern, two different correct
images, depending only on how far away you're looking, exactly like
looking at a near object versus a far one in real life.
"""

import torch
from angular_spectrum_propagation import AngularSpectrumPropagation
from phase_net import PhaseNet
from train_phase_net_depth import make_random_shape_batch_with_depth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wavelength = 532e-9
pixel_pitch = 8e-6
resolution = (256, 256)
z_near = 0.04
z_far = 0.06

propagator = AngularSpectrumPropagation(
    resolution=resolution, pixel_pitch=pixel_pitch, wavelength=wavelength, device=device
)
model = PhaseNet(in_channels=2).to(device)
model.load_state_dict(torch.load("phase_net_depth_weights.pt", map_location=device))
model.eval()

# unseen seed
generator = torch.Generator(device="cpu").manual_seed(999)
near_target, far_target, network_input = make_random_shape_batch_with_depth(
    1, resolution, device, generator
)

with torch.no_grad():
    raw_phase = model(network_input).squeeze(1)
    phase = torch.remainder(raw_phase, 2 * torch.pi)[0]
    field = torch.exp(1j * phase)

    recon_near = propagator(field, z_near).abs().cpu().numpy()
    recon_far = propagator(field, z_far).abs().cpu().numpy()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[0, 0].imshow(near_target[0].cpu().numpy(), cmap="gray")
axes[0, 0].set_title("Near target")
axes[0, 1].imshow(far_target[0].cpu().numpy(), cmap="gray")
axes[0, 1].set_title("Far target")
axes[1, 0].imshow(recon_near, cmap="gray")
axes[1, 0].set_title(f"Reconstruction @ z_near={z_near*1000:.0f}mm")
axes[1, 1].imshow(recon_far, cmap="gray")
axes[1, 1].set_title(f"Reconstruction @ z_far={z_far*1000:.0f}mm")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.savefig("depth_eval.png", dpi=150)
print("Saved depth_eval.png")
print("Check: bottom-left should resemble top-left (near), bottom-right should resemble top-right (far)")
print("If bottom-left instead resembles top-right, or vice versa, depth-selectivity failed.")
