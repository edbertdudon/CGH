"""
First step toward using real, non-procedural content for the eyebox/
depth-locking tests (see the eyebox-architecture work in Docs/..., and the
"translation v1" concept -- text that must appear locked to a real object's
depth, not floating at a fixed screen distance like current AR subtitles).

Takes an ordinary photo (no special depth-sensor hardware needed) and runs
a monocular depth-estimation neural network on it -- a model trained
specifically to guess per-pixel depth from a single 2D image. This is an
ESTIMATE, not a measurement, but is good enough for what this project needs
next: testing whether our multi-plane rendering can correctly lock content
to an object's real depth as the eyebox viewpoint shifts, using content
with actual geometry instead of flat hand-placed layers.

Model: Depth-Anything-V2-Small (via HuggingFace transformers' depth-
estimation pipeline) -- small and fast enough to run per-frame on a 3060
later if needed, good general-purpose relative-depth quality.

Usage:
    python3 estimate_depth.py path/to/your_photo.jpg
    (with no argument, generates and processes a synthetic placeholder
    photo instead, just to verify the pipeline runs end-to-end before a
    real photo is available)
"""
import sys
import numpy as np
from PIL import Image
import torch


def make_placeholder_photo(path="placeholder_room.png"):
    """A crude synthetic 'room with one object' photo -- NOT real content,
    only for verifying the depth pipeline runs before a real photo exists.
    A darker gradient 'background wall' with a bright rectangular 'object'
    closer to the camera, so a working depth model should read the
    rectangle as nearer than the gradient behind it."""
    h, w = 480, 640
    bg = np.tile(np.linspace(60, 160, w, dtype=np.uint8), (h, 1))
    img = np.stack([bg, bg, bg], axis=-1)
    img[150:330, 220:420] = [230, 200, 120]  # the "object" (e.g. a sign/book)
    Image.fromarray(img).save(path)
    print(f"No photo given -- wrote a synthetic placeholder to {path} "
          f"(pipeline-check only, not real content).")
    return path


def main():
    photo_path = sys.argv[1] if len(sys.argv) > 1 else make_placeholder_photo()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading Depth-Anything-V2-Small (first run downloads the model, ~100MB)...")
    from transformers import pipeline
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if device == "cuda" else -1,
    )

    image = Image.open(photo_path).convert("RGB")
    print(f"Input photo: {photo_path} ({image.size[0]}x{image.size[1]})")

    result = depth_pipe(image)
    depth = np.array(result["depth"])  # relative depth, higher = nearer (model convention)

    np.save("depth_map.npy", depth)
    result["depth"].save("depth_visualization.png")

    print(f"Depth map: shape={depth.shape}, min={depth.min():.1f}, max={depth.max():.1f}")
    print("Saved: depth_map.npy (raw array), depth_visualization.png (grayscale preview)")
    print("\nNote: this model's output is RELATIVE depth (nearer vs. farther, arbitrary")
    print("scale) not an absolute distance in meters -- fine for testing whether our")
    print("rendering correctly orders/locks content by depth, not for a real physical")
    print("distance measurement. If a real distance is later needed (e.g. to know the")
    print("actual eyebox/waveguide geometry to render for), it would need calibrating")
    print("against a known reference distance in the scene, or a real depth sensor.")


if __name__ == "__main__":
    main()
