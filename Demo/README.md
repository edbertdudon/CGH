# CGH Compute Validation — Starter Environment

## What this is

A runnable simulation of the phase-retrieval step described in Section 4.4
of the technical reference doc (the "14 FFTs/frame" estimate). It builds a
synthetic layered-3D target, runs multi-plane Gerchberg-Saxton phase
retrieval against it, and reports an *exact, measured* FFT count instead
of an assumed one — plus the image quality that count actually buys you.

It does **not** simulate the waveguide, the eye, real speckle perception,
or anything requiring supplier data. Those are still Gaps 1, 3, 4, 7 in
the doc, and still need the specialist conversations. This tool closes
the one gap that doesn't need anyone else's input first: whether the
compute estimate is in the right ballpark.

## What the first run already found

At reduced scale (256×256, 3 depth planes), plain multi-plane GS
**plateaued at ~11.5 dB PSNR after 7 iterations (84 FFT calls) and stopped
improving** — well short of a reasonable quality bar. This is a known
property of classic GS on hard-edged targets, not a bug: it stagnates
rather than converging further.

Two things worth noting immediately:

1. **The reconstructed images are visibly grainy/speckled** even where
   they're recognizable (see `convergence.png`). That graininess is not
   simulation noise — it's the same speckle problem Gap 7 in the doc
   names as "unsolved in the general case." This is the first time that
   gap has a visual, not just a bulleted, presence.
2. **This means the honest state of Gap 2 (compute) is: "14 FFTs/frame
   under plain GS is probably optimistic,"** at least until tested with
   (a) softer/anti-aliased target content instead of hard binary shapes,
   and (b) the gradient-based (SGD/Adam) alternative sketched in
   `retrieval.py::sgd_stub()`, which is standard in current CGH research
   and often needs fewer iterations for the same quality. That comparison
   — plain GS vs. SGD, same targets, same quality bar, exact FFT count
   each — is the single most useful next experiment.

Don't take the 11.5 dB number as a verdict on the whole architecture. Take
it as evidence that the compute estimate needs testing before it's quoted
to a silicon architect as settled, which is exactly the posture the doc
already commits to.

### Update: soft-edge targets (tested, still CPU/numpy)

Softening the target edges (Gaussian blur, `soft=True` in
`make_multiplane_target`) was tested against the stagnation hypothesis:
GS plateaus at **12.7 dB instead of 11.5 dB** — a real but modest
improvement. Hard edges were part of the problem, not most of it. The
bigger lever is the algorithm itself, which is what the torch/GPU files
below are for.

### Update: GPU port (torch) — written, not yet run

`propagation_torch.py`, `retrieval_torch.py`, `fft_counter_torch.py`, and
`demo_torch.py` port everything to torch and add a working SGD/Adam
phase-retrieval algorithm (previously just a stub) that optimizes SLM
phase directly via backprop through the same propagation model — standard
in current CGH research, often needing fewer FFTs than GS for equal
quality.

**These files were written and syntax-checked in an environment without a
GPU or torch installed, so they have not been run end-to-end.** The
propagation math was mirrored as closely as possible from the validated
numpy version (including a verified check that the coordinate-grid
convention matches exactly between `numpy.meshgrid` and
`torch.meshgrid(indexing="ij")`), but treat the first real run on your
3060 as the actual validation step, not this description. If something
throws an error, the traceback will point at exactly where the port
diverges — that's useful information, not a failure.

Run `demo_torch.py` first: it runs GS on GPU as a sanity check (should
reproduce roughly the same plateau behavior as the CPU run), then runs
SGD at equal FFT budget and compares directly, plus reports real
wall-clock timing.


## Files

- `fft_counter.py` / `fft_counter_torch.py` — shared counters; every
  FFT/IFFT call anywhere in the project increments them, so FFT counts
  are measured, not assumed.
- `propagation.py` / `propagation_torch.py` — angular spectrum method
  (depth-plane propagation). CPU/numpy version validated; torch/GPU
  version mirrors it, not yet run end-to-end (see above).
- `targets.py` — synthetic multi-depth-plane target generator, now with a
  `soft=True` option (Gaussian-blurred edges) that measurably reduces GS
  stagnation.
- `retrieval.py` / `retrieval_torch.py` — multi-plane GS (both, tested on
  CPU) plus the SGD/Adam alternative (torch only, GPU-required, not yet
  run).
- `metrics.py` — PSNR and a speckle-contrast proxy.
- `demo.py` — CPU/numpy sweep, prints the FFT-count/quality report, saves
  `convergence.png`. Already run — see results above.
- `demo_torch.py` — GPU: runs GS then SGD at equal FFT budget on the same
  targets, times both, plots a head-to-head convergence comparison. Run
  this on your 3060.

## Running it

CPU/numpy (already validated, no extra installs beyond numpy/scipy/matplotlib):
```
python3 demo.py
```

GPU/torch (run this on your machine with the 3060):
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
python3 demo_torch.py
```
(check https://pytorch.org/get-started/locally/ for the CUDA build that
matches your driver version)

## Scaling toward something you'd actually quote to a silicon architect

This is the real path from "toy demo" to "defensible number":

1. **Fix the algorithm first.** Get plain GS or SGD to actually converge
   to a real quality bar at small scale before scaling resolution — a
   bigger grid won't fix stagnation.
2. **Port to torch, run on GPU.** Swap `numpy.fft` for `torch.fft`, keep
   the same FFT counter pattern. This unlocks the SGD/Adam variant
   (needs autodiff) and makes full-resolution runs tractable.
3. **Scale resolution toward 4700×2700.** Expect wall-clock time to grow
   faster than pixel count (FFT cost is N·log N per axis); budget for it.
4. **Use real target content**, not synthetic shapes — actual layered
   renders from whatever engine will drive this, at realistic depth
   complexity, not 3 flat cartoon planes.
5. **Replace the placeholder depths.** `DEPTHS_M` in `demo.py` are
   illustrative separations scaled for the small demo aperture, not
   physically derived accommodation distances. At full resolution, derive
   them from the actual dioptric range you want to support.
6. **Once FFT-count-at-quality is solid, multiply by real GFLOP/FFT**
   (measured via `torch.profiler`, not the 5·N·log₂N estimate) to get a
   measured GFLOP/frame and TFLOP/s number — the one you'd actually bring
   to the silicon architect conversation in place of Section 4.4's
   estimate.

## Where this plugs back into the doc

- Section 4.4 (CGH compute derivation) — this is the thing to rerun and
  replace with a measured number before the silicon architect meeting.
- Section 4.5 (power) — inherits directly from 4.4; a revised FFT count
  changes the power gap multiplier too.
- Gap 2 (compute) in Section 5 — "plausible path via hardwired FFT block"
  is more defensible once backed by a measured FFT-count-at-quality, not
  an assumed one.
- Gap 7 (speckle) in Section 5 — now has a visual reference, not just a
  named risk.
