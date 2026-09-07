"""
Second real-content scene, for checking whether 10.13's finding (discrete-
tile wins on average quality on real content, reversing the near-tie found
on procedural content) generalizes beyond the first photo (bright airport
signage) or was specific to that content's concentrated bright regions.

This photo (IMG_2289, iPhone 13 Pro) is structurally different on purpose:
a wrought-iron fence very close to the camera, an etched (not backlit)
metal plaque mounted on it, and a church building + trees much further
back -- no bright glowing signage, a bigger foreground/background depth
gap, and a more textured, lower-contrast near-field subject (the ironwork).

No translation caption this time -- the caption step was specific to the
first photo's English-text region and isn't essential to testing whether
the eyebox-architecture result generalizes; keeping this build simple.

Run:
    python3 build_second_scene_target.py
"""
import numpy as np
from PIL import Image

SHAPE = (512, 512)
PHOTO_PATH = "IMG_2289_full.png"
DEPTH_PATH = "depth_map_2289.npy"
SUFFIX = "_2289"


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

    centers = kmeans_1d(depth.flatten()[::20])
    thresholds = (centers[:-1] + centers[1:]) / 2
    print(f"Depth cluster centers (far->near): {centers.round(1)}, thresholds: {thresholds.round(1)}")

    photo_sq, pad_x = pad_to_square(photo)
    depth_sq, _ = pad_to_square(depth)

    photo_r = resize(photo_sq, SHAPE, order=1)
    depth_r = resize(depth_sq, SHAPE, order=1)

    bucket = np.digitize(depth_r, thresholds)  # 0=far, 1=mid, 2=near

    near_arr = np.where(bucket == 2, photo_r, 0.0)
    mid_arr = np.where(bucket == 1, photo_r, 0.0)
    far_arr = np.where(bucket == 0, photo_r, 0.0)

    np.save(f"target_near{SUFFIX}.npy", near_arr)
    np.save(f"target_mid{SUFFIX}.npy", mid_arr)
    np.save(f"target_far{SUFFIX}.npy", far_arr)

    def to_img(a):
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

    to_img(near_arr).save(f"preview_near{SUFFIX}.png")
    to_img(mid_arr).save(f"preview_mid{SUFFIX}.png")
    to_img(far_arr).save(f"preview_far{SUFFIX}.png")

    composite = np.clip(np.stack([near_arr, mid_arr, far_arr], axis=-1), 0, 1)
    Image.fromarray((composite * 255).astype(np.uint8)).save(f"preview_composite{SUFFIX}.png")

    print(f"Saved target_near{SUFFIX}.npy, target_mid{SUFFIX}.npy, target_far{SUFFIX}.npy")
    print(f"Plane pixel fractions: near={np.mean(bucket==2)*100:.1f}%, mid={np.mean(bucket==1)*100:.1f}%, "
          f"far={np.mean(bucket==0)*100:.1f}%")


if __name__ == "__main__":
    main()
