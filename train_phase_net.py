"""
train_phase_net.py

Trains PhaseNet to predict phase-only SLM patterns in a single forward
pass, using the same validated AngularSpectrumPropagation layer as the
fixed physics inside the loop -- exactly the training-loop diagram from
earlier: Network -> ASM layer -> Loss, with gradients flowing back
through both, but only the network's weights actually get updated.

Uses a small procedurally-generated dataset of random circles rather
than an external dataset, so this runs with no downloads required. Swap
in a real image dataset (e.g. DIV2K, per the original roadmap) once this
is confirmed working -- the training loop itself doesn't change.
"""

import torch
from angular_spectrum_propagation import AngularSpectrumPropagation
from phase_net import PhaseNet


def make_random_shape_batch(batch_size, resolution, device, generator):
    """A tiny synthetic dataset: each image is 2-4 random circles."""
    ny, nx = resolution
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
    )
    batch = torch.zeros(batch_size, 1, ny, nx, device=device)
    for b in range(batch_size):
        n_shapes = torch.randint(2, 5, (1,), generator=generator).item()
        for _ in range(n_shapes):
            cy = torch.randint(40, ny - 40, (1,), generator=generator).item()
            cx = torch.randint(40, nx - 40, (1,), generator=generator).item()
            radius = torch.randint(10, 35, (1,), generator=generator).item()
            r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            batch[b, 0][r < radius] = 1.0
    return batch


def normalized(x, eps=1e-8):
    """Flatten-and-normalize each image in a batch to unit energy, so the
    loss compares shape, not absolute brightness scale -- same idea as
    the Gerchberg-Saxton error metric."""
    b = x.shape[0]
    flat = x.reshape(b, -1)
    norm = torch.linalg.vector_norm(flat, dim=1, keepdim=True) + eps
    return (flat / norm).reshape(x.shape)


def main():
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    generator = torch.Generator(device="cpu").manual_seed(0)
    batch_size = 8
    n_steps = 5000

    loss_history = []
    for step in range(n_steps):
        target = make_random_shape_batch(batch_size, resolution, device, generator)

        optimizer.zero_grad()
        raw_phase = model(target).squeeze(1)              # (B, H, W)
        phase = torch.remainder(raw_phase, 2 * torch.pi)  # exact, per the wrapping discussion
        field = torch.exp(1j * phase)

        reconstructed = propagator(field, z)  # our validated physics layer, unchanged
        recon_amp = reconstructed.abs()

        loss = torch.mean((normalized(recon_amp) - normalized(target.squeeze(1))) ** 2)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.6f}")

    print(f"Final loss: {loss_history[-1]:.6f}")
    torch.save(model.state_dict(), "phase_net_weights.pt")
    print("Saved phase_net_weights.pt")


if __name__ == "__main__":
    main()
