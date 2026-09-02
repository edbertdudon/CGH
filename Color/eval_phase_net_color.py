"""
eval_phase_net_color.py

Loads the trained color network, runs it on an unseen RGB target, and
combines the three separately-reconstructed color channels back into a
single actual color image -- the real test here is simply whether it
looks like a color picture, not three unrelated grayscale panels.
"""

import torch
import numpy as np
from angular_spectrum_propagation import AngularSpectrumPropagation
from phase_net import PhaseNet
from train_phase_net_color import make_rgb_batch, WAVELENGTHS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pixel_pitch = 8e-6
resolution = (256, 256)
z = 0.05

propagators = {
    c: AngularSpectrumPropagation(resolution, pixel_pitch, wl, device=device)
    for c, wl in WAVELENGTHS.items()
}

model = PhaseNet(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load("phase_net_color_weights.pt", map_location=device))
model.eval()

generator = torch.Generator(device="cpu").manual_seed(999)  # unseen
target_rgb = make_rgb_batch(1, resolution, device, generator)

with torch.no_grad():
    raw_phase = model(target_rgb)
    phase = torch.remainder(raw_phase, 2 * torch.pi)

    recon_channels = []
    for i, c in enumerate(["R", "G", "B"]):
        field = torch.exp(1j * phase[0, i])
        recon = propagators[c](field, z).abs()
        recon_channels.append(recon.cpu().numpy())

target_img = target_rgb[0].permute(1, 2, 0).cpu().numpy()
recon_img = np.stack(recon_channels, axis=-1)
recon_img = recon_img / recon_img.max()  # normalize for display

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(target_img)
axes[0].set_title("Target (color)")
axes[1].imshow(recon_img)
axes[1].set_title("Reconstruction (color)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("color_eval.png", dpi=150)
print("Saved color_eval.png")
