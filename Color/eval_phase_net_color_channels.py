"""
eval_phase_net_color_channels.py

Diagnostic: shows each of the three color channels' reconstruction
separately, in grayscale, instead of composited into one color image.
This directly checks whether the muddy coloring in the composite comes
from crosstalk -- each channel's reconstruction leaking brightness into
the other channels' spot locations -- the same phenomenon already seen
with depth, rather than something new.
"""

import torch
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

# same unseen test image as the color_eval.png run
generator = torch.Generator(device="cpu").manual_seed(999)
target_rgb = make_rgb_batch(1, resolution, device, generator)

with torch.no_grad():
    raw_phase = model(target_rgb)
    phase = torch.remainder(raw_phase, 2 * torch.pi)

    recons = {}
    for i, c in enumerate(["R", "G", "B"]):
        field = torch.exp(1j * phase[0, i])
        recons[c] = propagators[c](field, z).abs().cpu().numpy()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, c in enumerate(["R", "G", "B"]):
    axes[0, i].imshow(target_rgb[0, i].cpu().numpy(), cmap="gray")
    axes[0, i].set_title(f"{c} target")
    axes[1, i].imshow(recons[c], cmap="gray")
    axes[1, i].set_title(f"{c} channel reconstruction")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.savefig("color_channels_breakdown.png", dpi=150)
print("Saved color_channels_breakdown.png")
print("Check: does the R-channel reconstruction (bottom-left) light up ONLY at")
print("the R target's spot, or also faintly at the G and B spots?")
print("Lighting up at all three would confirm crosstalk between channels.")
