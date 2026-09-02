"""
train_phase_net_color.py

Extends the network to full color. Same pattern as depth: instead of
checking the reconstruction against one target at one setting, we check
it against three targets at three settings -- here, three wavelengths
instead of two distances. Each wavelength needs its own phase map (a
phase-only SLM can't do all three colors from one pattern), so the
network now outputs three independent phase channels instead of one.

Three separate AngularSpectrumPropagation instances are used, one per
wavelength, each with its own transfer function -- this mirrors a real
three-panel full-color SLM setup, the option flagged as an alternative
to time-multiplexing clear back in the first message of this
conversation.
"""

import torch
from angular_spectrum_propagation import AngularSpectrumPropagation
from phase_net import PhaseNet

# Standard-ish RGB laser wavelengths for holography
WAVELENGTHS = {"R": 638e-9, "G": 532e-9, "B": 450e-9}


def make_rgb_batch(batch_size, resolution, device, generator):
    """Independent random circle content per color channel -- a harder,
    more honest test than just recoloring identical shapes three times."""
    ny, nx = resolution
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
    )
    channels = []
    for _ in range(3):
        chan = torch.zeros(batch_size, ny, nx, device=device)
        for b in range(batch_size):
            for _ in range(torch.randint(1, 3, (1,), generator=generator).item()):
                cy = torch.randint(30, ny - 30, (1,), generator=generator).item()
                cx = torch.randint(30, nx - 30, (1,), generator=generator).item()
                radius = torch.randint(10, 25, (1,), generator=generator).item()
                r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                chan[b][r < radius] = 1.0
        channels.append(chan)
    return torch.stack(channels, dim=1)  # (B, 3, H, W)


def normalized(x, eps=1e-8):
    b = x.shape[0]
    flat = x.reshape(b, -1)
    norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True) + eps
    return (flat / norm).reshape(x.shape)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    pixel_pitch = 8e-6  # same physical SLM, all three wavelengths
    resolution = (256, 256)
    z = 0.05

    # one propagator per wavelength -- the "three panels" approach
    propagators = {
        c: AngularSpectrumPropagation(resolution, pixel_pitch, wl, device=device)
        for c, wl in WAVELENGTHS.items()
    }

    model = PhaseNet(in_channels=3, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    generator = torch.Generator(device="cpu").manual_seed(0)
    batch_size = 8
    n_steps = 5000

    loss_history = []
    for step in range(n_steps):
        target_rgb = make_rgb_batch(batch_size, resolution, device, generator)

        optimizer.zero_grad()
        raw_phase = model(target_rgb)  # (B, 3, H, W)
        phase = torch.remainder(raw_phase, 2 * torch.pi)

        loss = 0.0
        channel_losses = {}
        for i, c in enumerate(["R", "G", "B"]):
            field = torch.exp(1j * phase[:, i])
            recon = propagators[c](field, z).abs()
            ch_loss = torch.mean((normalized(recon) - normalized(target_rgb[:, i])) ** 2)
            channel_losses[c] = ch_loss.item()
            loss = loss + ch_loss

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step % 250 == 0:
            print(
                f"step {step:4d}  loss {loss.item():.6f}  "
                f"(R {channel_losses['R']:.6f}, G {channel_losses['G']:.6f}, B {channel_losses['B']:.6f})"
            )

    print(f"Final loss: {loss_history[-1]:.6f}")
    torch.save(model.state_dict(), "phase_net_color_weights.pt")
    print("Saved phase_net_color_weights.pt")


if __name__ == "__main__":
    main()
