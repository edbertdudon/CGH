"""
train_phase_net_depth.py

Extends the single-plane network to real depth. The target now has
content at two different distances (near and far), encoded the way
RGB-D would be -- one channel for what's there, one channel for how far
away it is. The network still outputs a single phase pattern, but that
one pattern now has to reconstruct correctly at TWO different
propagation distances at once: near content in focus at z_near, far
content in focus at z_far. That's the actual definition of a 3D
hologram, rather than a flat 2D image glued to one fixed distance.
"""

import torch
from angular_spectrum_propagation import AngularSpectrumPropagation
from phase_net import PhaseNet


def make_random_shape_batch_with_depth(batch_size, resolution, device, generator):
    """
    Returns:
        near_target, far_target : (B, H, W) -- content at each depth, on
            its own, for supervision.
        network_input : (B, 2, H, W) -- channel 0 is combined content
            (near + far together), channel 1 is a depth map (1.0 at
            near-content pixels, 0.0 at far-content pixels).
    """
    ny, nx = resolution
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij"
    )
    near_target = torch.zeros(batch_size, ny, nx, device=device)
    far_target = torch.zeros(batch_size, ny, nx, device=device)

    for b in range(batch_size):
        for _ in range(torch.randint(1, 3, (1,), generator=generator).item()):
            cy = torch.randint(30, ny - 30, (1,), generator=generator).item()
            cx = torch.randint(30, nx - 30, (1,), generator=generator).item()
            radius = torch.randint(10, 25, (1,), generator=generator).item()
            r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            near_target[b][r < radius] = 1.0
        for _ in range(torch.randint(1, 3, (1,), generator=generator).item()):
            cy = torch.randint(30, ny - 30, (1,), generator=generator).item()
            cx = torch.randint(30, nx - 30, (1,), generator=generator).item()
            radius = torch.randint(10, 25, (1,), generator=generator).item()
            r = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            far_target[b][r < radius] = 1.0

    content = torch.clamp(near_target + far_target, max=1.0)
    depth_map = torch.zeros_like(content)
    depth_map[near_target > 0] = 1.0

    network_input = torch.stack([content, depth_map], dim=1)  # (B, 2, H, W)
    return near_target, far_target, network_input


def normalized(x, eps=1e-8):
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
    z_near = 0.04  # 40mm
    z_far = 0.06   # 60mm

    propagator = AngularSpectrumPropagation(
        resolution=resolution, pixel_pitch=pixel_pitch, wavelength=wavelength, device=device
    )
    model = PhaseNet(in_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    generator = torch.Generator(device="cpu").manual_seed(0)
    batch_size = 8
    n_steps = 5000

    loss_history = []
    for step in range(n_steps):
        near_target, far_target, network_input = make_random_shape_batch_with_depth(
            batch_size, resolution, device, generator
        )

        optimizer.zero_grad()
        raw_phase = model(network_input).squeeze(1)
        phase = torch.remainder(raw_phase, 2 * torch.pi)
        field = torch.exp(1j * phase)

        recon_near = propagator(field, z_near).abs()
        recon_far = propagator(field, z_far).abs()

        loss_near = torch.mean((normalized(recon_near) - normalized(near_target)) ** 2)
        loss_far = torch.mean((normalized(recon_far) - normalized(far_target)) ** 2)
        loss = loss_near + loss_far

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if step % 250 == 0:
            print(
                f"step {step:4d}  loss {loss.item():.6f}  "
                f"(near {loss_near.item():.6f}, far {loss_far.item():.6f})"
            )

    print(f"Final loss: {loss_history[-1]:.6f}")
    torch.save(model.state_dict(), "phase_net_depth_weights.pt")
    print("Saved phase_net_depth_weights.pt")


if __name__ == "__main__":
    main()
