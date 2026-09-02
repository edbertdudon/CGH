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


def make_realistic_multiplane_target(shape):
    """
    Three content types, one per depth plane, chosen to be both a
    stronger generalization test than the synthetic disc/ring/checker
    shapes AND directly representative of the travel-translation use
    case (world-locked content at real-world depth):

      near:  a UI icon (navigation arrow) -- sharp graphic content
      mid:   translated caption text -- thin strokes, small features;
             known to be a hard case for holographic reconstruction,
             harder than any synthetic shape tested so far
      far:   a procedurally generated photo-like scene -- gradual
             shading and texture, closer to natural image statistics
             than a flat blurred blob

    All content is generated procedurally (rendered text, drawn
    vector-style icon, synthetic texture) rather than sourced from real
    photographs, both to sidestep any copyright question and to give
    precise, reproducible control over content difficulty.
    """
    from PIL import Image, ImageDraw, ImageFont
    ny, nx = shape

    # ---- near: UI navigation icon ----
    near = Image.new("L", (nx, ny), 0)
    d = ImageDraw.Draw(near)
    cx, cy = nx / 2, ny / 2
    s = min(nx, ny) * 0.22
    # simple arrow/chevron, drawn as filled polygon
    pts = [
        (cx, cy - s), (cx + s * 0.7, cy + s * 0.3), (cx + s * 0.25, cy + s * 0.3),
        (cx + s * 0.25, cy + s), (cx - s * 0.25, cy + s), (cx - s * 0.25, cy + s * 0.3),
        (cx - s * 0.7, cy + s * 0.3),
    ]
    d.polygon(pts, fill=255)
    near_arr = np.array(near, dtype=np.float64) / 255.0

    # ---- mid: translated caption text ----
    # Rendered via matplotlib, not PIL+system-font-path: PIL's truetype
    # loader depends on a font file existing at a specific OS path, which
    # is not portable across machines. A first version of this hardcoded
    # a Debian-specific path that silently fell back to PIL's tiny,
    # non-scaling default bitmap font on other systems -- producing
    # near-invisible text and a misleadingly high PSNR (a mostly-black
    # target is trivially easy to "match"). Matplotlib bundles its own
    # font (DejaVu Sans) and works the same on any machine that already
    # has matplotlib, which every script in this project requires anyway.
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(nx / 100, ny / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text = "STAZIONE CENTRALE"
    ax.text(0.5, 0.5, text, color="white", fontsize=ny * 0.05, ha="center", va="center",
            fontweight="bold", transform=ax.transAxes)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    mid_arr = buf[:, :, :3].mean(axis=2) / 255.0
    if mid_arr.shape != (ny, nx):
        # canvas rounding can be off by a pixel or two; crop/pad to match exactly
        from scipy.ndimage import zoom
        mid_arr = zoom(mid_arr, (ny / mid_arr.shape[0], nx / mid_arr.shape[1]), order=1)

    # ---- far: procedurally generated photo-like scene (gradient sky +
    # horizon + soft shapes, giving natural-image-like gradual shading) ----
    y, x = np.mgrid[0:ny, 0:nx]
    horizon = ny * 0.6
    sky = np.clip(1.0 - (y / horizon), 0, 1) ** 0.7
    ground = np.clip((y - horizon) / (ny - horizon), 0, 1) * 0.5
    scene = np.where(y < horizon, sky * 0.7, ground)
    # a couple of soft "building" silhouettes for extra spatial structure
    rng = np.random.default_rng(0)
    for _ in range(4):
        bx = rng.uniform(0.1, 0.9) * nx
        bw = rng.uniform(0.05, 0.15) * nx
        bh = rng.uniform(0.15, 0.35) * ny
        mask = (np.abs(x - bx) < bw / 2) & (y > horizon - bh) & (y < horizon)
        scene[mask] = 0.35
    from scipy.ndimage import gaussian_filter
    far_arr = gaussian_filter(scene, sigma=min(nx, ny) * 0.004)
    far_arr = far_arr / (far_arr.max() + 1e-8)

    return [near_arr, mid_arr, far_arr]


def sparse_content_bounds(target, margin_frac=0.15):
    """
    Bounding box (as row/col slices) of the non-zero content in a sparse
    target -- caption text, a UI icon, anything that's mostly empty
    background -- padded by margin_frac of the box's own size. Use this
    to compute a masked/local PSNR on sparse content: whole-frame PSNR
    is dominated by trivially correct background pixels and can look
    deceptively high (or hide a real quality difference between
    conditions) even when the actual content reconstructs poorly.
    Content that fills the frame (like a photo-like scene with no large
    empty background) doesn't need this -- whole-frame PSNR is already
    meaningful there.
    """
    rows = np.where(target.max(axis=1) > 0.05)[0]
    cols = np.where(target.max(axis=0) > 0.05)[0]
    if len(rows) == 0 or len(cols) == 0:
        return slice(0, target.shape[0]), slice(0, target.shape[1])
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    rm = int((r1 - r0) * margin_frac)
    cm = int((c1 - c0) * margin_frac)
    r0 = max(0, r0 - rm)
    r1 = min(target.shape[0], r1 + rm)
    c0 = max(0, c0 - cm)
    c1 = min(target.shape[1], c1 + cm)
    return slice(r0, r1), slice(c0, c1)


# kept as an alias -- earlier version of this module used the text-specific name
text_region_bounds = sparse_content_bounds
