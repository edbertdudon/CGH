"""
Builds this project's first non-procedural, real-photo multi-plane CGH
target -- replacing the icon/text/photo synthetic content used everywhere
else with a real airport photo (IMG_2443, iPhone 13 Pro) plus its
AI-estimated depth map (estimate_depth.py), and a Japanese translation
caption locked to the REAL estimated depth of the sign it translates.

Depth bucketing: a quick 1D k-means over the whole depth map found 3
natural clusters at ~57 (far/background), ~137 (mid), ~204 (near) on the
model's 0-255 relative scale. The "Baggage Reclaim" sign's own sampled
depth (median 137 over its plate region) lands almost exactly on the mid
cluster center, and the electronic flight/bag-status board's sampled
depth (median 198) lands almost exactly on the near cluster center --
both consistent with the user's own visual read (the board looks nearer;
the sign is mounted further back on the pillar structure). Thresholds at
the cluster midpoints (97.1, 170.9) are used to assign every pixel to
one of 3 depth planes, matching this project's existing 3-plane
(near/mid/far) convention so this content can drop into the same CGH
pipeline used everywhere else.

Caption: "手荷物受取所" (baggage claim, standard Japanese airport signage
term) rendered in a real Japanese font (Windows-bundled MS Gothic --
matplotlib's bundled DejaVu Sans has no CJK glyphs, so this step is
Windows-specific, unlike the rest of this project's portable target
generator), composited into the MID plane at the same screen position
the original English "Baggage Reclaim" text occupies in the photo --
i.e. locked to the real depth of the sign it's translating, not floating
at an arbitrary fixed depth the way a normal AR caption would.

Run:
    python3 build_realistic_target.py
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SHAPE = (512, 512)
PHOTO_PATH = "IMG_2443_full.png"
DEPTH_PATH = "depth_map.npy"

# Sign region in the ORIGINAL 3024x4032 photo (pixel box containing the
# full "Baggage Reclaim <-- 1-10" sign plate) and the English text's own
# sub-region within it, both found by visual inspection of crops.
SIGN_BOX = (300, 380, 2850, 780)          # (x0, y0, x1, y1)
ENGLISH_TEXT_BOX = (1189, 532, 2713, 710)  # (x0, y0, x1, y1), original coords

JAPANESE_CAPTION = "手荷物受取所"
JAPANESE_FONT_PATH = r"C:\Windows\Fonts\msgothic.ttc"


def pad_to_square(arr, fill=0):
    h, w = arr.shape[:2]
    if w == h:
        return arr, 0
    if w < h:
        pad = (h - w) // 2
        pad_shape = ((0, 0), (pad, h - w - pad)) + ((0, 0),) * (arr.ndim - 2)
    else:
        pad = (w - h) // 2
        pad_shape = ((pad, w - h - pad), (0, 0)) + ((0, 0),) * (arr.ndim - 2)
    return np.pad(arr, pad_shape, constant_values=fill), pad


def resize(arr, shape, order=1):
    from scipy.ndimage import zoom
    zy, zx = shape[0] / arr.shape[0], shape[1] / arr.shape[1]
    return zoom(arr, (zy, zx) + (1,) * (arr.ndim - 2), order=order)


def kmeans_1d(x, k=3, iters=50):
    centers = np.quantile(x, np.linspace(0.1, 0.9, k))
    for _ in range(iters):
        d = np.abs(x[:, None] - centers[None, :])
        assign = d.argmin(axis=1)
        new_centers = np.array([x[assign == i].mean() if np.any(assign == i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return np.sort(centers)


def main():
    photo = np.array(Image.open(PHOTO_PATH).convert("L"), dtype=np.float64) / 255.0
    depth = np.load(DEPTH_PATH).astype(np.float64)
    print(f"Loaded photo {photo.shape}, depth {depth.shape}")

    x0, y0, x1, y1 = SIGN_BOX
    sign_depth = depth[y0:y1, x0:x1]
    print(f"Sign-plate sampled depth: median={np.median(sign_depth):.1f}")

    centers = kmeans_1d(depth.flatten()[::20])
    thresholds = (centers[:-1] + centers[1:]) / 2
    print(f"Depth cluster centers (far->near): {centers.round(1)}, thresholds: {thresholds.round(1)}")

    # pad width up to height (or vice versa) with black, preserving all content, no cropping
    photo_sq, pad_x = pad_to_square(photo)
    depth_sq, _ = pad_to_square(depth)
    print(f"Padded to square: {photo_sq.shape} (pad_x={pad_x})")

    photo_512 = resize(photo_sq, SHAPE, order=1)
    depth_512 = resize(depth_sq, SHAPE, order=1)

    bucket = np.digitize(depth_512, thresholds)  # 0=far, 1=mid, 2=near

    near_arr = np.where(bucket == 2, photo_512, 0.0)
    mid_arr = np.where(bucket == 1, photo_512, 0.0)
    far_arr = np.where(bucket == 0, photo_512, 0.0)

    # map the English text's bounding box through the same pad+resize transform
    scale = SHAPE[0] / photo_sq.shape[0]
    ex0 = int((ENGLISH_TEXT_BOX[0] + pad_x) * scale)
    ex1 = int((ENGLISH_TEXT_BOX[2] + pad_x) * scale)
    ey0 = int(ENGLISH_TEXT_BOX[1] * scale)
    ey1 = int(ENGLISH_TEXT_BOX[3] * scale)
    print(f"English text region mapped to working resolution: x=({ex0},{ex1}) y=({ey0},{ey1})")

    caption_img = Image.new("L", (SHAPE[1], SHAPE[0]), 0)
    d = ImageDraw.Draw(caption_img)
    box_h = ey1 - ey0
    font = ImageFont.truetype(JAPANESE_FONT_PATH, size=max(8, int(box_h * 0.9)))
    bbox = d.textbbox((0, 0), JAPANESE_CAPTION, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
    d.text((cx - tw / 2 - bbox[0], cy - th / 2 - bbox[1]), JAPANESE_CAPTION, fill=255, font=font)
    caption_arr = np.array(caption_img, dtype=np.float64) / 255.0

    # composite: caption replaces the sign's own real content in that region of the mid plane,
    # locked to the sign's real depth (mid plane), not an arbitrary fixed depth
    mid_with_caption = mid_arr.copy()
    replace_mask = caption_arr > 0.05
    # clear the original English-text pixels in a slightly generous box, then stamp the caption
    mid_with_caption[max(0, ey0 - 4):ey1 + 4, max(0, ex0 - 4):ex1 + 4] = 0.0
    mid_with_caption = np.maximum(mid_with_caption, caption_arr)

    np.save("target_near.npy", near_arr)
    np.save("target_mid_with_caption.npy", mid_with_caption)
    np.save("target_mid_original.npy", mid_arr)
    np.save("target_far.npy", far_arr)

    def to_img(a):
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

    to_img(near_arr).save("preview_near.png")
    to_img(mid_with_caption).save("preview_mid_with_caption.png")
    to_img(far_arr).save("preview_far.png")

    # colorized composite: near=red, mid=green, far=blue, for a single at-a-glance preview
    composite = np.stack([near_arr, mid_with_caption, far_arr], axis=-1)
    composite = np.clip(composite, 0, 1)
    Image.fromarray((composite * 255).astype(np.uint8)).save("preview_composite_rgb.png")

    print("\nSaved: target_near.npy, target_mid_with_caption.npy, target_mid_original.npy, target_far.npy")
    print("Saved previews: preview_near.png, preview_mid_with_caption.png, preview_far.png, preview_composite_rgb.png")
    print(f"\nPlane pixel fractions: near={np.mean(bucket==2)*100:.1f}%, mid={np.mean(bucket==1)*100:.1f}%, far={np.mean(bucket==0)*100:.1f}%")


if __name__ == "__main__":
    main()
