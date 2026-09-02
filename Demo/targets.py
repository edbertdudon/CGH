"""
Synthetic multi-depth-plane targets.

Standing in for a real layered-3D scene (which would come from a game
engine / renderer feeding depth-segmented layers). Shapes are chosen so
image quality and per-plane focus separation are both easy to eyeball.
"""
import numpy as np


def make_target_plane(shape, kind="disc"):
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    cy, cx = ny / 2, nx / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if kind == "disc":
        img = (r < min(ny, nx) * 0.18).astype(float)
    elif kind == "ring":
        img = ((r > min(ny, nx) * 0.28) & (r < min(ny, nx) * 0.36)).astype(float)
    elif kind == "checker":
        block = max(ny, nx) // 16
        img = (((x // block) + (y // block)) % 2).astype(float)
        # confine to a central patch so planes don't fully overlap
        mask = (np.abs(x - cx) < nx * 0.3) & (np.abs(y - cy) < ny * 0.3)
        img = img * mask
    else:
        raise ValueError(kind)
    return img


def make_multiplane_target(shape, n_planes=3, soft=False, sigma=2.0):
    kinds = ["disc", "ring", "checker"]
    planes = [make_target_plane(shape, kind=kinds[i % len(kinds)]) for i in range(n_planes)]
    if soft:
        from scipy.ndimage import gaussian_filter
        planes = [gaussian_filter(p, sigma=sigma) for p in planes]
    return planes
