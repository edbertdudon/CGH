# Holographic AR at Glasses Scale: Compute, Power, Eyebox Architecture, and Image-Quality Limits of Multi-Plane Phase-Only CGH

**Author:** Edbert Dudon ([@edbertdudon](https://github.com/edbertdudon))  
**Status:** Working technical report · companion document v31 · not peer reviewed  
**Companion document:** [AR Glasses Technical Reference v31](Docs/AR_Glasses_Technical_Reference_v31.docx)

---

## Abstract

Current near-eye displays hold focus at a fixed optical distance. This produces a vergence–accommodation conflict (VAC) that limits comfortable wear. Varifocal and multifocal designs reduce the conflict, but they require actuation that does not fit in a glasses frame. Computer-generated holography (CGH) removes VAC by construction. Whether its computation fits a glasses power envelope is an open question.

We derive a target specification from the space-bandwidth product: 4,700 × 2,700 phase pixels at 2.0 µm pitch, a 50° × 28° field of view, and a 5.0 mm effective eyebox. At that full resolution we measure three-plane Gerchberg–Saxton phase retrieval with angular-spectrum propagation. Every run is taken to plateau-verified convergence across 8 random seeds. Convergence takes 2,618 FFTs per solve on average (range 865–3,592). With the eyebox covered by four time-multiplexed tiles at 360 Hz, and a 16-bit block-floating-point accelerator at 6.5 TOPS/W, this implies ~942 W, about 1,900× a 500 mW budget.

Two specification-level changes reduce this figure:
- **Relaxed angular resolution.** Moving to 1.2 arcmin per pixel, near the limit of human acuity and matching a published waveguide holography system, cuts power to ~248 W with no measured quality loss.
- **Joint eyebox pattern.** Replacing tile cycling with a single phase pattern jointly optimized over a continuum of eye positions removes discrete snapping and lowers the required frame rate from 360 Hz to 90 Hz. Each joint solve needs ~3.4× more FFTs, so at the relaxed resolution it costs ~212 W (~423× budget). That is only modestly below the ~248 W of four cycled tiles.

Reconstruction quality is limited by what a single phase-only pattern can represent across several depths, not by the optimizer; four independent lines of evidence point this way. Quantization emulation finds 8-bit precision matching float32 (+0.01 dB), a potential further ~3× power reduction pending hardware confirmation. We also test two real photographs at 512 × 512, each with a monocular depth estimate and a translated caption locked to the sign's depth. Quality is 9.5 dB and 7.9 dB, and the better eyebox strategy flips between the two scenes. Finally, we document the findings that reversed under multi-seed, full-resolution, or verified-convergence re-testing, and the measurement rules those reversals produced.

**Keywords:** computer-generated holography · holographic near-eye display · vergence–accommodation conflict · Gerchberg–Saxton · angular spectrum method · synthetic aperture · power budget · quantization · reproducibility

---

## Headline results

| Quantity | Value | Evidence |
| --- | --- | --- |
| FFTs per solve, 3 planes, 4,700 × 2,700 | 2,618 (865–3,592) | 8 seeds, 400-iteration budget |
| Power, 4 cycled tiles at 360 Hz, full resolution (BFP16) | ~942 W (311–1,293 W), ~1,884× budget | 8 seeds |
| Same, relaxed resolution (~1.2 arcmin/px) | ~248 W, ~497× budget; quality +0.49 dB | 8 seeds |
| Joint eyebox pattern at 90 Hz, relaxed resolution | ~212 W (83–354 W), ~423× budget | 8 seeds, 800-step budget |
| Joint + INT8, if confirmed on hardware | ~69 W, ~138× budget | Quantization emulation only |
| 8-bit vs. float32 quality | +0.01 dB | Full resolution, true convergence |
| Masked reconstruction quality, procedural scene | ~7.0 dB (full res), ~7.5 dB (relaxed) | 8 seeds |
| Masked reconstruction quality, real photos (512 × 512) | 9.53 dB and 7.92 dB | 5 seeds each |
| Sensitivity to optimizer schedule | 0.50 dB spread across 4 schedules | 3 seeds × 4, full resolution |
| Warm-start speedup, small inter-frame motion | ~8–9× (7.67–9.33×) | 3 seeds, full resolution |
| Sequential vs. simultaneous depth planes | Sequential better in 25 of 25 seed pairings | Full resolution |
| Color: one joint pattern vs. per-color solve | −1.64 dB | 3 seeds, full resolution |

---

## 1. Introduction

Conventional near-eye displays present imagery at a fixed focal distance, while stereo disparity drives the eyes to converge at varying distances. The resulting vergence–accommodation conflict degrades visual performance and causes fatigue [1, 2]. Varifocal and multifocal designs reduce its severity but do not eliminate it, and their actuators are hard to package in an eyeglass frame.

Holographic displays avoid the conflict by reconstructing a wavefront, so the eye accommodates to each point as it would to a real scene [5, 7]. Recent systems have demonstrated compact holographic hardware built on waveguides, steered illumination, and learned propagation models [8, 9]. Holography also turns prescription correction into extra phase terms in the same computation, so no corrective inserts are needed.

These advantages cost computation. A phase-only spatial light modulator (SLM) must be driven by a hologram computed from the target scene, usually by iterative phase retrieval, fast enough to also cover the eyebox. This work asks a narrow question: at the pixel count, frame rate, depth structure, and eyebox a glasses-form device needs, how much computation does phase retrieval take, what image quality does it reach, and how far is that from a wearable power and thermal envelope?

This report makes five contributions:

1. It derives a complete target specification from the space-bandwidth product and expresses power directly in terms of measured FFT counts (Section 2).
2. It measures convergence cost and reconstruction quality of three-plane Gerchberg–Saxton (GS) at the full 4,700 × 2,700 target, using seed-averaged, plateau-verified convergence (Section 4.1).
3. It tests two specification-level levers: relaxed angular resolution, and a jointly optimized eyebox pattern in place of time-multiplexed tiles (Sections 4.2–4.3). It also tests several algorithmic levers: numeric precision, temporal warm-starting, initialization, sequential and reduced depth-plane rendering, and color multiplexing (Sections 4.6–4.12).
4. It moves from procedural test scenes to real photographs with estimated depth and a depth-locked translated caption (Section 4.5).
5. It records every finding that reversed under stricter re-testing, and the measurement rules those reversals produced (Section 5).

## 2. System model

### 2.1 Space-bandwidth product

Every spatial parameter follows from the space-bandwidth product, which is conserved through any lossless passive optic. The required phase-pixel count along one axis is

$$N = \frac{2 \, W_{\text{eyebox}} \, \sin(\text{FOV}/2)}{\lambda}$$

where $W_{\text{eyebox}}$ is the instantaneous eyebox width, FOV is the full field of view on that axis, and $\lambda$ is the shortest (worst-case) wavelength. With $\lambda = 450$ nm, $W_{\text{eyebox}} = 2.5$ mm, and half-angles of 25° and 14°, this gives $N_x = 4{,}696$ and $N_y = 2{,}688$, about 12.6 Mpix.

Pixel pitch does not appear in the equation. Pitch sets the physical panel size and the angular magnification the waveguide must supply, not the pixel count.

### 2.2 Derived specification

| Parameter | Target | Basis |
| --- | --- | --- |
| Phase resolution | 4,700 × 2,700 (~12.6 Mpix, ~0.63 arcmin/px) | Space-bandwidth product at 450 nm |
| Relaxed resolution (recommended, not yet formally adopted) | ~2,475 × 1,422 (~3.5 Mpix, ~1.2 arcmin/px) | Human acuity; demonstrated system [9] |
| Pixel pitch | 2.0 µm | Sets panel size (9.4 × 5.4 mm at full resolution); no confirmed supplier |
| Native diffraction half-angle | ±6.46° | $\sin\theta = \lambda / 2p$ |
| Required waveguide angular magnification | ~3.87× | 25° ÷ 6.46° |
| Field of view | 50° × 28° (~57° diagonal) | Design target |
| Eyebox | 2.5 mm instantaneous, 5.0 mm effective | Joint pattern (adopted) or 2 × 2 cycled tiles |
| Frame rate | 90 Hz with the joint eyebox pattern; 360 Hz with 4 cycled tiles; ×3 for field-sequential color | Perceptual base rate × patterns per frame |
| Power budget | ≤ 500 mW | Glasses form factor |
| Sustainable thermal load | ~324–540 mW | 10.8 cm² skin contact at 30–50 mW/cm² |
| Laser optical output | ~18 mW | 3,000 nits at 0.2% system efficiency |

### 2.3 From FFT count to watts

Phase-retrieval cost is dominated by 2D FFTs. Sustained power is

$$P = \frac{n \cdot c \cdot f}{\eta}$$

where:
- $n$ is the number of FFTs per solve;
- $c$ is the cost of one FFT at the padded array size (6.50 GFLOP at full resolution, 1.67 GFLOP at the relaxed resolution);
- $f$ is the number of solves per second (360 for four tiles each shown at 90 Hz, 90 for a single joint pattern);
- $\eta$ is accelerator efficiency (6.5 TOPS/W for BFP16, 20 TOPS/W for INT8).

**Full resolution, four tiles.** One FFT costs $6.5 \times 10^{9} / 6.5 \times 10^{12} = 1$ mJ. At 360 solves per second, each FFT in a solve adds ~0.36 W. A single GS iteration uses 12 FFTs (measured totals scale at 12 per iteration). That one iteration therefore costs ~4.3 W, before convergence has even started.

**Relaxed resolution, joint pattern, 90 Hz.** Each FFT in a solve adds only ~23 mW at BFP16, so the 500 mW budget affords about 21 FFTs per solve. One joint optimization step uses 24 FFTs (a batch of 4 viewpoints). At BFP16 a single step therefore costs ~556 mW, just over budget. At INT8 the same step costs ~181 mW, so about 2.8 steps fit within budget.

A joint solve that converges in two or three steps, for example through warm-starting from the previous frame, would be within reach of the power envelope. That regime has not yet been tested.

## 3. Methods

### 3.1 Forward model and phase retrieval

Light is propagated from the SLM plane to three target depth planes (near, mid, far; synthetic proxy distances of 1, 3, and 6 mm) with the angular spectrum method [4]. Multi-plane GS [3] runs as follows:

1. Propagate the current phase-only field to each plane.
2. At each plane, replace the amplitude with the target amplitude and keep the phase.
3. Back-propagate each plane's field to the SLM and average them.
4. Keep only the phase of the average.

Comparison methods are direct pixel-wise gradient descent (Adam) and a CNN-parameterized ("neural-prior") phase representation. The pipeline is written in PyTorch and runs on an NVIDIA RTX 3060, with FFTs counted by an instrumented counter rather than a formula. Before use, the propagation layer was validated in isolation: a lens-focusing test placed the focal distance within 1.19% of the expected value, and energy was conserved to a ratio of 1.0000.

### 3.2 Eyebox strategies

Three strategies for covering the eyebox are compared:

- **Discrete tiles:** N independent patterns, each optimized for one fixed eye position and cycled in time.
- **Joint:** a single pattern optimized against a batch of 4 eye positions sampled at random from the whole eyebox range at every step. This follows the structure of the subaperture objective in [9], without that system's physical waveguide model.
- **Local-robust hybrid:** a few patterns, each optimized against positions sampled from its own sub-region.

Viewpoint changes are modeled with a light-field stand-in: each depth layer is shifted sideways in proportion to $z_{\text{far}}/z$, so nearer layers move more. This is a simplified parallax model, not a 4D light-field renderer or a physical pupil model. Quality is evaluated on 15 held-out viewpoints spanning the range.

### 3.3 Test content

**Procedural scene.** Sparse content sits at the near and mid planes (a UI icon and translated caption text), and a dense photo-like scene sits at the far plane. Supplementary procedural content includes:
- purpose-built sparse and dense mid-plane profiles;
- independently generated scenes;
- constant-velocity pans;
- a parallax motion model.

**Real content.** Two iPhone 13 Pro photographs are used:
- **Photo 1:** bilingual airport baggage-reclaim signage with an electronic status board.
- **Photo 2:** a wrought-iron fence and etched plaque in front of a church and trees.

The photos have no sensor depth. Relative depth is estimated with Depth-Anything-V2-Small [10], and every pixel is bucketed into near, mid, and far planes by 1D k-means. For Photo 1, a Japanese translation of "Baggage Reclaim" is composited into the mid plane at the sign's own position, so it renders at the sign's estimated depth rather than at a fixed screen distance. Depth ordering is real, but the propagation distances remain the synthetic proxies above.

### 3.4 Convergence detection

Runs use generous budgets: 400 iterations for GS and 800 steps for the joint-pattern baseline. Convergence is declared with a sustained-hold plateau detector, and a measurement is accepted only if the run is not still rising when its budget ends. When convergence speed is compared between conditions, both conditions must meet this standard.

### 3.5 Seeds and scale

Initial phase is random. Results are reported as means over multiple seeds, with ranges: 8 seeds for compute baselines and 3–7 elsewhere. Results obtained at 512 × 512 are treated as provisional until they are repeated at full resolution.

### 3.6 Quality metrics

Quality is PSNR in intensity: the reconstructed amplitude is squared before being compared with target intensity. Sparse planes are scored only inside a content mask; otherwise, trivially correct background inflates the score.
- Compact content (an icon or a caption) uses a bounding box.
- Real photos, whose depth planes are scattered in disconnected pieces, use a per-pixel mask.

Some convergence-speed experiments use an amplitude-domain metric that matches GS's internal objective. Comparisons are made within an experiment, not across experiments scored differently. No perceptual quality threshold has been established.

### 3.7 Quantization emulation

To emulate reduced precision, the complex field is rounded to $N$-bit fixed point:
- after each forward propagation;
- after the amplitude constraint;
- after each backward propagation.

The quantization scale is calibrated once and then held static. All other arithmetic stays in float32. Rounding error inside an FFT's internal butterfly stages is not modeled.

## 4. Results

### 4.1 Compute baseline at full resolution

These results use three-plane GS at 4,700 × 2,700, with 8 seeds and a 400-iteration budget. No seed was still rising when its budget ended.

| Seed | Plateau iteration | FFTs/solve | Power (4 tiles, 360 Hz) | × budget |
| --- | --- | --- | --- | --- |
| 0 | 260 | 3,124 | 1,124 W | 2,248× |
| 1 | 268 | 3,220 | 1,159 W | 2,317× |
| 2 | 72 | 865 | 311 W | 623× |
| 3 | 198 | 2,379 | 856 W | 1,712× |
| 4 | 222 | 2,667 | 960 W | 1,920× |
| 5 | 299 | 3,592 | 1,293 W | 2,585× |
| 6 | 249 | 2,992 | 1,076 W | 2,153× |
| 7 | 175 | 2,103 | 757 W | 1,513× |
| **Mean** | | **2,618** | **~942 W** | **~1,884×** |

An earlier 5-seed measurement with a 100-iteration budget reported 552 FFTs per solve. That figure understated the cost about 4×. Inspecting full convergence histories showed the reason: some seeds keep improving well past iteration 100 (one gained 0.38 dB between iterations 65 and 220), and others pass through sustained dips lasting over 100 iterations before settling. The spread across seeds is a property of the algorithm, not an artifact of measurement. A 1–2 Wh battery would sustain the mean draw for roughly 4–8 seconds.

### 4.2 Relaxing angular resolution

The full-resolution target works out to ~0.63 arcmin per pixel. That is nearly twice as fine as 20/20 human acuity (~1 arcmin) and finer than the 1.2 arcmin demonstrated in [9]. Both resolutions below were measured with 8 seeds, a 400-iteration budget, and padding held fixed.

| Resolution | FFTs/solve | Per-FFT cost | Power (4 tiles, 360 Hz) | Masked quality |
| --- | --- | --- | --- | --- |
| 4,700 × 2,700 (~0.63 arcmin/px) | 2,618 (865–3,592) | 6.50 GFLOP | ~942 W (311–1,293 W) | 6.99 dB (6.82–7.15) |
| ~2,475 × 1,422 (~1.2 arcmin/px) | 2,684 (1,394–3,665) | 1.67 GFLOP | ~248 W (129–339 W) | 7.48 dB (7.33–7.59) |

Iteration count is essentially unchanged (103%). The 74% power reduction comes entirely from cheaper FFTs on the smaller array, and quality is slightly higher, not lower.

The same test was repeated on Photo 1 with its fine caption text, at a reduced scale of 1024² versus 512². Power fell 58% and quality rose 0.61 dB. On this content, however, the smaller array needed 86% more FFTs to converge. The saving held only because each FFT cost about a quarter as much.

The lever is therefore supported on both content types. Relaxing resolution is still a specification decision rather than a software optimization, because it accepts less angular resolution than Section 2.1 derives for the stated eyebox and field of view.

### 4.3 Eyebox architecture

All comparisons in this section were run at 512 × 512 with 6 seeds, 500 optimization steps, and a wide eyebox range, unless noted otherwise. The "tile" column shows the best-matching discrete tile at each viewpoint. "Std" is the variation in quality across viewpoints, where lower means more uniform.

| Content and budget | Joint (dB) | Tile (dB) | Joint wins | Joint std / tile std |
| --- | --- | --- | --- | --- |
| Procedural, narrow range, 4 tiles | — | tile ahead by ~0.14 | 2/6 | — |
| Procedural, 4 tiles | 7.30 | 7.28 | 4/6 | 0.09 / 0.17 |
| Procedural, 2 tiles | 7.30 | 7.18 | 6/6 | 0.09 / 0.22 |
| Photo 1 (airport signage), 4 tiles | 9.10 | 9.19 | 1/6 | 0.09 / 0.15 |
| Photo 2 (fence, plaque, church), 4 tiles | 8.08 | 7.76 | 6/6 | 0.11 / 0.11 |
| Hybrid, procedural, 2 local-robust positions | 7.41 | — | — | 0.13 |

The pattern across rows:
- On procedural content, discrete tiling becomes less uniform as each tile covers more territory, while the joint pattern's uniformity stays flat.
- The difference in average quality is small, and its direction depends on content: it reverses between the two real photos.
- A naive "flicker" score, which averages the reconstructions of all cycled tiles, appeared to favor tiling. Averaging four patterns trained on the *same* viewpoint produced the same +2.37 dB gain, so that score reflects generic speckle averaging and was excluded.

**Decision.** Because the quality data does not settle the question, the joint pattern was adopted on structural grounds. Discrete tiles snap between fixed eye positions by construction; a single joint pattern has no positions to snap between. The joint pattern needs one solve per displayed frame, so the required frame rate drops from 360 Hz to 90 Hz. Against the 60 Hz ceiling of current commercial monochrome phase SLMs, that is a gap of 1.5× instead of 6×.

**Joint cost baseline.** At the relaxed resolution, with 8 seeds and an 800-step budget (no seed still rising), the joint pattern averaged 9,141 FFTs per solve (range 3,576–15,312). Held-out quality was 7.00 dB (6.87–7.09).

<!-- NOTE (v31 doc): the companion document states the joint-vs-tile reduction as ~4.6×, using ~245 W for one tile at 90 Hz and ~980 W for four. By the Section 2.3 formula, one tile at 90 Hz is ~62 W and four cycled tiles are ~248 W, which is the figure already in Section 4.2. The table below uses the formula-consistent values. -->

| Eyebox strategy (relaxed resolution) | Solves/s | FFTs/solve | Power (BFP16) | × budget |
| --- | --- | --- | --- | --- |
| 4 discrete tiles | 360 | 2,684 | ~248 W | ~497× |
| Joint pattern | 90 | 9,141 | ~212 W (83–354 W) | ~423× |
| Joint pattern, INT8 (if confirmed) | 90 | 9,141 | ~69 W | ~138× |

The joint pattern needs 4× fewer solves per second but ~3.4× more FFTs per solve. The net power reduction is therefore about 15%. Its main gains are the absence of snapping and the lower frame-rate requirement.

**Hybrid pilot.** Two positions, each optimized over its own ±5 px neighborhood, reached 7.41 dB, above both pure strategies, with uniformity between them. It still has some snapping, and its FFT cost has not been measured.

### 4.4 Image-quality ceiling

Four lines of evidence indicate that the quality ceiling reflects what one phase-only pattern can represent, not a failure of the search:

1. **Averaging independent solutions** gives only a small gain, so much of the error is systematic.
2. **More simultaneous planes lowers quality, though not in a clean three-step ladder.** These results use masked scoring at full resolution. Going from 2 planes (7.40 dB, 5 seeds) to 3 planes (7.00 dB, 3 seeds) is a consistent drop: 9 of 9 seed pairings agree. One plane (7.84 dB, 7 seeds) cannot be separated from two: the pairwise win rate stayed near a coin flip as seeds were added.
3. **Optimizer choice barely matters.** At full resolution, with 3 seeds per schedule and 800 steps, four schedules differ by 0.50 dB overall:

| Schedule | Seeds (dB) | Mean (dB) |
| --- | --- | --- |
| Fixed lr = 0.02 | 11.95, 11.63, 11.68 | 11.75 |
| Fixed lr = 0.05 | 11.84, 11.95, 12.13 | 11.97 |
| Cosine decay | 11.66, 11.96, 11.71 | 11.78 |
| One-cycle | 11.85, 11.67, 11.81 | 11.78 |

4. **A CNN-parameterized phase does not help.** At full resolution, its quality peaks early and then collapses to a flat, lower plateau as optimization continues (for example, 8.18 dB at step 29 falling to 6.34 dB from step ~100 onward). Across 3 seeds, the best checkpoint of each run averaged 8.80 dB. Each training step also costs ~10× a gradient-descent step.

The schedule sweep and the precision results in Section 4.6 were scored with an earlier setup, which reads ~11–12 dB for the same three-plane scene that the current masked scoring places at ~7.0 dB. Differences within those experiments are meaningful; their absolute values should not be compared with the masked figures.

### 4.5 Real content

On Photo 1, bounding-box masking first reported 14.94 dB with a suspiciously small spread across seeds (std 0.01 dB). The bounding boxes covered 65–97% of the frame, but only 16–44% of each box held content, so empty background was inflating the score. With a per-pixel content mask, quality is 9.53 dB (5 seeds, 800-iteration budget, every seed plateaued). The reconstructed caption remains legible, with visibly more noise.

Photo 2, with finer and lower-contrast near-field detail, reaches 7.92 dB (5 seeds).

### 4.6 Numeric precision

| Precision | Final PSNR (dB), full resolution, true convergence | Δ vs. float32 |
| --- | --- | --- |
| 4-bit | 10.79 | −1.06 |
| 8-bit | 11.86 | +0.01 |
| 16-bit | 11.94 | +0.09 |
| float32 | 11.85 | — |

The 8-bit result survived three attempts to break it:
- replacing an oracle quantization scale with a static calibrated one;
- a full-resolution rerun to true convergence;
- a multi-seed check that traced an apparent "8-bit needs more iterations" effect to ordinary seed variance.

At 4 bits, quality degrades and plateaus within a few iterations.

### 4.7 Sequential vs. simultaneous depth planes

These results use 5 seeds per condition (25 seed pairings) at full resolution and are scored with the amplitude-domain metric.

| Plane | Simultaneous mean (dB) | Sequential mean (dB) | Δ range across 25 pairings |
| --- | --- | --- | --- |
| Near (sparse) | 7.47 | 8.79 | +0.87 to +2.06 |
| Mid (sparse) | 10.53 | 11.21 | +0.30 to +1.21 |
| Far (dense) | 11.23 | 14.70 | +2.81 to +4.42 |

Sequential rendering never lost in any pairing. It does cost three sub-frames per frame. For sparse content, the gain is too small to justify that cost.

### 4.8 Color

These results are at full resolution and true convergence, with 3 seeds, averaged over red and blue across all three planes.

| Approach | PSNR (dB) | Δ vs. native | Sub-frames |
| --- | --- | --- | --- |
| Native (separate solve per color) | 9.97 (7.32–12.55) | — | 3 |
| Joint (one pattern, 9 constraints) | 8.33 (7.10–9.70) | −1.64 | 1, same total FFT cost as native |
| Naive reuse (green phase, unmodified) | 6.97 (6.69–7.28) | −3.00 | 1 |

### 4.9 Temporal warm-starting

Here each GS solve starts from the previous frame's converged phase instead of from random noise. Speedup is the reduction in iterations needed to reach threshold quality.

| Motion | 512 × 512 | 4,700 × 2,700 |
| --- | --- | --- |
| Steady pan, ~0.6% of frame | ~14.9× | ~8–9× (7.67–9.33×, 3 seeds) |
| Steady pan, ~12% of frame | ~8.3× | ~9× mean (5.33–13.67×, 3 seeds) |

Averaged over seeds, the speedup falls smoothly with shift size, from 14.2× at 3 px to 6.3× at 150 px (512 × 512).

Across scene cuts, including full permutations of which content sits at which depth, no trial showed negative transfer at either scale.

On independently generated content:
- all 4 scene-cut pairs converged faster with warm-starting;
- under parallax motion, 17 of 21 frame-to-frame comparisons favored warm-starting.

These results lower average power over a session, not peak power. Warm-starting has not yet been tested with the joint eyebox pattern.

### 4.10 Dropping the middle depth plane

Here the solve targets only the near and far planes. On every profile and at every scale tested, the resulting field still reconstructs the mid plane well above a blank field. Two claims did not hold up:

- **Plane count chosen by content sparsity.** The ranking of sparse versus dense mid-plane profiles inverts between 512 × 512 and full resolution, so this idea is retracted.
- **Compute savings.** On the standard content at full resolution (3 seeds), two planes cost 28% *more* FFTs on average. Across the nine seed pairings, the effect ranged from a 32% saving to a 125% increase.

### 4.11 Initialization strategies (negative results)

Three deterministic alternatives to random initial phase were tested:

- **Flat-phase back-propagation.** It did worse than a zero-phase start.
- **Coarse-to-fine seeding.** Starting from a low-resolution solve did not reduce total FFT cost.
- **Donor phase.** Reusing a phase already converged on unrelated content looked 24–64% faster under a fixed budget. At verified convergence, it had no effect in one direction and cost 15% more iterations in the other.

None of the three reduces the seed-dependent variance in Section 4.1.

### 4.12 Padding factor

Lowering the padding factor showed compute savings of anywhere from 10% to 53%, depending on the number of seeds. It has not yet been measured with 8 seeds and a 400-iteration budget, so no single figure is reported.

## 5. Findings revised under re-testing

| Original claim | After re-testing | What changed the answer |
| --- | --- | --- |
| 14 FFTs/frame at 1.49 GFLOP (closed form) | Measured cost far higher | Measured convergence; profiling at padded size |
| 696 FFTs/frame, ~250 W | Not reproducible | Experiment state never committed |
| 434 FFTs/solve (1 seed) → 552 (5 seeds) | 2,618 FFTs/solve, ~942 W | 8 seeds, 400-iteration budget, stricter detector |
| Plane count: 14.6 / 11.9 / 11.5 dB | 7.84 / 7.40 / 7.00 dB; 1-vs-2 step unresolved | Full resolution, masked scoring, more seeds |
| Neural prior: unverified at full resolution | Collapses under continued optimization | Full resolution, 3 seeds |
| Real-photo quality 14.94 dB | 9.53 dB | Per-pixel mask instead of bounding box |
| Real content favors discrete tiles | Reversed by a second photo | Second real scene |
| Joint pattern ahead at wide range (+0.09 dB, 1 seed) | +0.01 dB, noise-level | 6 seeds |
| Tiles win on "flicker" score | Speckle-averaging artifact (+2.37 dB with no viewpoint difference) | Same-viewpoint control |
| Iteration count independent of resolution | True for procedural content only | Real content |
| Donor-phase initialization 24–64% faster | No effect, or 15% slower | Verified convergence |
| Catastrophic negative transfer after scene cuts | 0 of 10 trials | Verified convergence |
| Warm start converging almost instantly | 2 of 7 spot-checked frames were false positives | Plateau check against final value |
| 8-bit needs more iterations | Ordinary seed variance | Multiple seeds |
| No sequential-rendering benefit for sparse content | Positive in 25 of 25 pairings | Multiple seeds |
| Plane count can follow content sparsity | Ranking inverts; retracted | Full resolution |
| Joint color −0.5 dB, naive reuse −1.0 dB | −1.64 dB and −3.00 dB | True convergence, 3 seeds |
| Large-shift warm-start speedup ~6.0× | 5.33–13.67× | 3 seeds |
| Non-monotonic shift-sweep curve | Smooth decay | 3 seeds per point |
| Dropping mid plane saves ~23% compute | +28% cost on average; sign unstable | Full resolution, 3 seeds |

Five measurement rules follow from these revisions:

1. **Verified convergence.** Convergence claims require a plateau confirmed against where the run actually settles, within a budget long enough that the run is no longer rising. Short budgets flipped comparisons in both directions and hid a ~4× cost understatement. Detectors that check only local flatness reported false convergence.
2. **Multiple seeds.** Results are multi-seed means with ranges.
3. **Full resolution.** Small-scale results are provisional until repeated at full resolution.
4. **Masks match content.** A bounding box that fits a compact icon inflates scores on scattered real content.
5. **Reproducible experiment states.** Experiment code is committed when it is run, and the Python environment is pinned alongside any log that a reported figure depends on.

## 6. Limitations

**Physical modeling.** The simulation has no waveguide, eye, MEMS, or partial-coherence model. The eyebox comparisons use a lateral-shift parallax stand-in, and the correspondence between the simulated viewpoint range and a physical 5 mm eyebox has not been established. Propagation distances are synthetic proxies.

**Quality threshold.** PSNR values are not tied to a perceptual threshold. A human or optical-engineering judgment is still needed to decide what quality is sufficient.

**Scale.** The eyebox comparisons and both real-photo tests were run at 512 × 512. The real-content resolution check was run at 1024² versus 512², not at the target resolutions.

**Evidence strength.**
- Only two real photographs have been tested, and no recorded video.
- The joint pattern's learning rate and batch size have not been tuned.
- The hybrid pilot covers a single configuration.
- Several comparisons rest on 3 seeds, which bounds a range without pinning down a mean.
- The 8-bit result is algorithm-level emulation.

**Metrics and artifacts.** Absolute dB values from differently scored experiments are not directly comparable (Sections 3.6 and 4.4). A faint axis-aligned grid artifact in full-resolution reconstructions remains unexplained.

## 7. Future work

- **Padding factor:** an 8-seed measurement with a 400-iteration budget, to replace the current 10–53% range.
- **Joint and hybrid tuning:** tune the joint pattern's hyperparameters; sweep the hybrid's position count and neighborhood width; measure the hybrid's FFT cost.
- **Joint warm-starting:** test warm-started joint solves, which Section 2.3 identifies as the regime closest to the power envelope.
- **Full-resolution eyebox tests,** with more real scenes, recorded motion, and calibrated physical distances.
- **Fixed-point hardware:** a butterfly-accurate fixed-point FFT model, then FPGA validation of INT8.
- **Image quality:** perceptual testing to set a quality threshold, and double-phase and complex-amplitude encodings as further tests of the quality ceiling.
- **Hardware calibration:** a real SLM with camera-in-the-loop optimization [6].

## 8. Repository layout

<!-- TODO: confirm folder descriptions and add any folders not listed -->

```
CGH/
├── Docs/           # Technical reference (current: v31)
├── Reference/      # Third-party papers used for comparison (see License)
├── Demo/           # Core GS phase retrieval and angular-spectrum propagation
├── Neural-prior/   # CNN-parameterized phase representation
├── Compute/        # End-to-end convergence sweeps with FFT counting, output logs
├── Video/          # Temporal warm-starting, scene cuts, independent content, parallax motion
├── Eyebox/         # Discrete-tile vs. joint eyebox experiments (multi-seed)
├── Translation/    # Real photos, depth estimation, depth-locked captions, joint baseline, hybrid pilot
└── README.md
```

Scripts named in the companion document include:

| Result | Script |
| --- | --- |
| Compute baseline reconciliation | `experiment_end_to_end_padding_reconcile.py`, `experiment_detector_diagnostic.py` |
| Independent content and parallax motion | `Video/experiment_independent_content.py` |
| Eyebox, multi-seed and fewer tiles | `Eyebox/experiment_subaperture_wide_eyebox_multiseed.py`, `Eyebox/experiment_subaperture_fewer_tiles_multiseed.py` |
| Depth estimation and real-scene targets | `Translation/estimate_depth.py`, `Translation/build_second_scene_target.py` |
| Resolution relaxation, real content | `Translation/experiment_realistic_resolution_relaxation.py` |
| Joint eyebox cost baseline | `Translation/experiment_joint_true_baseline.py` |
| Local-robust hybrid pilot | `Translation/experiment_hybrid_local_robust.py` |

## 9. Reproducing results

**Requirements:** Python with PyTorch and CUDA, on an RTX 3060-class GPU (12 GB) or better. Full-resolution runs are memory-intensive; the neural-prior runs peaked at 10.4 GB. The depth-estimation step uses the Hugging Face `transformers` pipeline.

<!-- TODO: add pinned requirements.txt and exact run commands -->

```bash
git clone https://github.com/edbertdudon/CGH.git
cd CGH
pip install -r requirements.txt   # TODO: pin with pip freeze
python Translation/experiment_joint_true_baseline.py   # example; see Section 8
```

When reporting a new figure:
- give the mean across seeds with its range;
- confirm that no run was still rising when its budget ended;
- in any comparison, confirm that both conditions reached a verified plateau.

## 10. Citation

If you use this work, please cite it:

```bibtex
@misc{dudon2026holographic,
  author       = {Edbert Dudon},
  title        = {Holographic {AR} at Glasses Scale: Compute, Power, Eyebox
                  Architecture, and Image-Quality Limits of Multi-Plane
                  Phase-Only {CGH}},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Working technical report; companion document v31},
  url          = {https://github.com/edbertdudon/CGH}
}
```

## References

1. Hoffman, D. M., Girshick, A. R., Akeley, K., & Banks, M. S. (2008). Vergence–accommodation conflicts hinder visual performance and cause visual fatigue. *Journal of Vision*, 8(3), 33.
2. Kramida, G. (2016). Resolving the vergence-accommodation conflict in head-mounted displays. *IEEE Transactions on Visualization and Computer Graphics*, 22(7).
3. Gerchberg, R. W., & Saxton, W. O. (1972). A practical algorithm for the determination of phase from image and diffraction plane pictures. *Optik*, 35, 237–246.
4. Goodman, J. W. *Introduction to Fourier Optics*. Roberts & Company.
5. Maimone, A., Georgiou, A., & Kollin, J. S. (2017). Holographic near-eye displays for virtual and augmented reality. *ACM Transactions on Graphics*, 36(4).
6. Peng, Y., Choi, S., Padmanaban, N., & Wetzstein, G. (2020). Neural holography with camera-in-the-loop training. *ACM Transactions on Graphics*, 39(6).
7. Shi, L., Li, B., Kim, C., Kellnhofer, P., & Matusik, W. (2021). Towards real-time photorealistic 3D holography with deep neural networks. *Nature*, 591, 234–239.
8. Gopakumar, M., Lee, G.-Y., Choi, S., Chao, B., Peng, Y., Kim, J., & Wetzstein, G. (2024). Full-colour 3D holographic augmented-reality displays with metasurface waveguides. *Nature*, 629, 791–797.
9. Choi, S., Jang, C., Lanman, D., & Wetzstein, G. (2025). Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue. *Nature Photonics*, 19, 854–863.
10. Yang, L., et al. (2024). Depth Anything V2. *Advances in Neural Information Processing Systems (NeurIPS)*.

## License

Code and documentation in this repository are released under the [MIT License](LICENSE) © 2026 Edbert Dudon. You may use, modify, and redistribute them freely, provided the copyright notice is kept. If this work informs yours, please cite it (Section 10).

Third-party material in `Reference/`, including the Choi et al. paper, is not covered by this license and remains under its publisher's terms.
