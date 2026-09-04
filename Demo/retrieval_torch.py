"""
Two phase-retrieval algorithms, both on GPU via torch:

  multiplane_gs_torch   -- same alternating-projection algorithm as
                            retrieval.py, ported to torch. Run this first
                            as a sanity check: it should reproduce the CPU
                            numpy results (same plateau behavior) before
                            you trust anything else in this file.

  multiplane_sgd        -- the algorithm that was left as sgd_stub() in
                            retrieval.py. Optimizes SLM phase directly via
                            gradient descent (Adam) against an amplitude-
                            matching loss, using autodiff through the same
                            propagation model. This is the one worth
                            comparing against GS for the "how many FFTs
                            does it actually take" question.

Both report history in PSNR (dB), same metric as the CPU run, so results
are directly comparable to what you already have.
"""
import math
import torch
from propagation_torch import angular_spectrum_propagate, safe_abs
from fft_counter_torch import counter


def _to_tensor_targets(target_planes, device):
    return [
        torch.sqrt(torch.clamp(torch.tensor(t, dtype=torch.float32, device=device), min=0))
        for t in target_planes
    ]


def _psnr_torch(recon_amp, target_amp, eps=1e-8):
    r = recon_amp / (recon_amp.max() + eps)
    t = target_amp / (target_amp.max() + eps)
    mse = torch.mean((r - t) ** 2)
    if mse < eps:
        return 99.0
    return float(10 * torch.log10(1.0 / mse))


def make_edge_taper(shape, taper_width, device, dtype):
    """
    Smooth 2D raised-cosine taper: 1.0 in the interior, ramping down to
    0.0 over `taper_width` (fraction of each axis) at the edges. Used as
    the proposed fix from Section 10.4 -- constrains the SLM field at the
    frame borders instead of leaving them completely unconstrained, which
    is where GS was found to accumulate uncontested energy over
    iterations.
    """
    ny, nx = shape

    def taper_1d(n, width):
        w = max(1, int(n * width))
        t = torch.ones(n, device=device, dtype=dtype)
        ramp = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, w, device=device, dtype=dtype)))
        t[:w] = ramp
        t[-w:] = ramp.flip(0)
        return t

    ty = taper_1d(ny, taper_width)
    tx = taper_1d(nx, taper_width)
    return ty.unsqueeze(1) * tx.unsqueeze(0)


def multiplane_gs_torch(target_planes, depths_m, wavelength, dx, n_iters, device="cuda", seed=0, pad_factor=2,
                         complex_dtype=torch.complex64, smooth_cutoff=False, cutoff_width=0.05,
                         edge_taper=False, taper_width=0.05, init_phase=None):
    """
    Returns (slm_phase, history, recon_by_checkpoint) -- same shape as the
    CPU version in retrieval.py, so plotting code can treat both the same
    way.

    FFT-count note: quality is measured from the forward pass already
    computed inside the main loop (field_at_plane), not a separate extra
    pass -- this matches the CPU version's approach and means GS's
    reported FFT count is the real algorithmic cost, not inflated by
    instrumentation. (An earlier version of this file added a second,
    unnecessary forward pass just to measure PSNR, adding 6 phantom FFT
    calls per iteration -- that's why GS looked more expensive relative to
    SGD than it actually was. Fixed here.)

    History indexing note (found via temporal warm-start testing): the
    in-loop `field_at_plane` forward pass reflects the phase BEFORE that
    iteration's correction is applied, not after -- an earlier version of
    this function scored quality from that pre-correction pass and
    labeled it as the iteration's result, silently recording each
    iteration's *starting* quality as its *ending* quality. That shifted
    every entry in `history` back by one iteration and meant the very
    last correction applied (at it=n_iters) was never scored at all --
    `history[-1]` (used everywhere as "final quality") was actually the
    quality after n_iters-1 corrections. Fixed by scoring recon_planes at
    the top of the NEXT iteration (or one extra forward-only pass after
    the loop, for the last iteration) instead of the current one. This
    costs one extra forward pass per plane, once, at the very end --
    negligible next to the per-iteration budget, and necessary for
    history to mean what it claims.
    """
    shape = target_planes[0].shape
    targets_amp = _to_tensor_targets(target_planes, device)
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32

    taper = None
    if edge_taper:
        taper = make_edge_taper(shape, taper_width, device, real_dtype)

    if init_phase is not None:
        slm_phase = init_phase.to(device=device, dtype=real_dtype)
    else:
        torch.manual_seed(seed)
        slm_phase = (torch.rand(shape, device=device, dtype=real_dtype) * 2 * math.pi - math.pi)
    slm_field = torch.exp(1j * slm_phase)
    if taper is not None:
        slm_field = slm_field * taper.to(complex_dtype)

    history = []
    checkpoints = sorted(set([1, max(1, n_iters // 2), n_iters]))
    recon_by_checkpoint = {}

    for it in range(1, n_iters + 1):
        correction = torch.zeros(shape, dtype=complex_dtype, device=device)
        recon_planes = []
        for target_amp, z in zip(targets_amp, depths_m):
            field_at_plane = angular_spectrum_propagate(slm_field, wavelength, dx, z, pad_factor=pad_factor,
                                                          complex_dtype=complex_dtype, smooth_cutoff=smooth_cutoff,
                                                          cutoff_width=cutoff_width)
            recon_planes.append(safe_abs(field_at_plane).detach())
            phase_at_plane = torch.angle(field_at_plane)
            constrained = target_amp.to(complex_dtype) * torch.exp(1j * phase_at_plane.to(real_dtype))
            back = angular_spectrum_propagate(constrained, wavelength, dx, -z, pad_factor=pad_factor,
                                               complex_dtype=complex_dtype, smooth_cutoff=smooth_cutoff,
                                               cutoff_width=cutoff_width)
            correction += back

        # recon_planes above reflects the phase BEFORE this iteration's
        # correction -- i.e. the quality after (it-1) corrections, not it.
        # Score it as iteration (it-1)'s result, not this one.
        if it > 1:
            psnrs = [_psnr_torch(r, t) for r, t in zip(recon_planes, targets_amp)]
            history.append(sum(psnrs) / len(psnrs))
            if (it - 1) in checkpoints:
                recon_by_checkpoint[it - 1] = [r.cpu().numpy() for r in recon_planes]

        slm_field = correction / len(depths_m)
        slm_phase = torch.angle(slm_field)
        slm_field = torch.exp(1j * slm_phase)
        if taper is not None:
            slm_field = slm_field * taper.to(complex_dtype)

    # One final forward-only pass to score the last iteration's correction
    # (n_iters), which the loop above computes but never measures.
    final_recon_planes = []
    for target_amp, z in zip(targets_amp, depths_m):
        field_at_plane = angular_spectrum_propagate(slm_field, wavelength, dx, z, pad_factor=pad_factor,
                                                      complex_dtype=complex_dtype, smooth_cutoff=smooth_cutoff,
                                                      cutoff_width=cutoff_width)
        final_recon_planes.append(safe_abs(field_at_plane).detach())
    psnrs = [_psnr_torch(r, t) for r, t in zip(final_recon_planes, targets_amp)]
    history.append(sum(psnrs) / len(psnrs))
    if n_iters in checkpoints:
        recon_by_checkpoint[n_iters] = [r.cpu().numpy() for r in final_recon_planes]

    return slm_phase.detach(), history, recon_by_checkpoint


def multiconstraint_gs_torch(constraints, dx, n_iters, device="cuda", seed=0, pad_factor=2,
                              complex_dtype=torch.complex64):
    """
    Generalizes multiplane_gs_torch: instead of one shared wavelength
    across several depth planes, each constraint carries its own
    (target, z, wavelength). Same GS logic either way -- propagate to
    each constraint, replace amplitude, propagate back, average the
    corrections, re-enforce phase-only. This makes "solve N depth planes
    at M colors jointly, in one phase pattern" the same algorithm as
    ordinary multi-plane GS, just with more constraints in the list.

    constraints: list of dicts, each {"target": 2D array (amplitude,
    values >=0), "z": depth in meters, "wavelength": meters}
    """
    shape = constraints[0]["target"].shape
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    targets_amp = []
    for c in constraints:
        t = torch.tensor(c["target"], dtype=real_dtype, device=device)
        targets_amp.append(torch.sqrt(torch.clamp(t, min=0)))

    torch.manual_seed(seed)
    slm_phase = (torch.rand(shape, device=device, dtype=real_dtype) * 2 * math.pi - math.pi)
    slm_field = torch.exp(1j * slm_phase)

    history = []
    checkpoints = sorted(set([1, max(1, n_iters // 2), n_iters]))
    recon_by_checkpoint = {}

    for it in range(1, n_iters + 1):
        correction = torch.zeros(shape, dtype=complex_dtype, device=device)
        recon_list = []
        for c, target_amp in zip(constraints, targets_amp):
            field_at = angular_spectrum_propagate(slm_field, c["wavelength"], dx, c["z"],
                                                    pad_factor=pad_factor, complex_dtype=complex_dtype)
            recon_list.append(safe_abs(field_at).detach())
            phase_at = torch.angle(field_at)
            constrained = target_amp.to(complex_dtype) * torch.exp(1j * phase_at.to(real_dtype))
            back = angular_spectrum_propagate(constrained, c["wavelength"], dx, -c["z"],
                                               pad_factor=pad_factor, complex_dtype=complex_dtype)
            correction += back

        slm_field = correction / len(constraints)
        slm_phase = torch.angle(slm_field)
        slm_field = torch.exp(1j * slm_phase)

        psnrs = [_psnr_torch(r, t) for r, t in zip(recon_list, targets_amp)]
        history.append(sum(psnrs) / len(psnrs))
        if it in checkpoints:
            recon_by_checkpoint[it] = [r.cpu().numpy() for r in recon_list]

    return slm_phase.detach(), history, recon_by_checkpoint


def multiplane_sgd(target_planes, depths_m, wavelength, dx, n_steps, lr=0.02, device="cuda", seed=0,
                    lr_schedule=None):
    """
    Optimizes SLM phase directly by gradient descent on an amplitude-
    matching loss, backpropagating through the same angular-spectrum
    propagation model used by GS. This is standard practice in current
    CGH research (differentiable wave propagation + Adam) and typically
    needs fewer FFT evaluations than plain GS for the same quality --
    that comparison is the point of running both.

    lr_schedule: optional torch.optim.lr_scheduler class or None. Fixed
    lr can undershoot early (slow start) or overshoot late (stalls near
    the optimum instead of settling into it) -- a schedule that starts
    higher and decays is standard practice for exactly this symptom.
    Pass e.g. "cosine" to use CosineAnnealingLR(T_max=n_steps), or
    "onecycle" for OneCycleLR. None keeps the original fixed-lr behavior.

    FFT-count caveat: the counter only sees forward-pass FFT calls.
    autograd's backward() runs its own FFT-like operations internally to
    compute gradients through torch.fft -- those are real compute cost but
    invisible to fft_counter_torch. So SGD's reported FFT count is a real
    undercount of its true cost (rough rule of thumb: backward through an
    FFT costs about the same as the FFT itself, so true cost is roughly
    ~2x the forward count reported here). Keep this in mind when comparing
    against GS's now-accurate count.
    """
    shape = target_planes[0].shape
    targets_amp = _to_tensor_targets(target_planes, device)

    torch.manual_seed(seed)
    slm_phase = (torch.rand(shape, device=device) * 2 * math.pi - math.pi).requires_grad_(True)
    opt = torch.optim.Adam([slm_phase], lr=lr)

    scheduler = None
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    elif lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr * 3, total_steps=n_steps)

    history = []
    lr_history = []
    checkpoints = sorted(set([1, max(1, n_steps // 2), n_steps]))
    recon_by_checkpoint = {}

    for step in range(1, n_steps + 1):
        opt.zero_grad()
        field = torch.exp(1j * slm_phase)

        loss = torch.tensor(0.0, device=device)
        psnrs = []
        recon_planes = []
        for target_amp, z in zip(targets_amp, depths_m):
            recon = angular_spectrum_propagate(field, wavelength, dx, z)
            recon_amp = safe_abs(recon)
            loss = loss + torch.mean((recon_amp - target_amp) ** 2)
            psnrs.append(_psnr_torch(recon_amp.detach(), target_amp))
            recon_planes.append(recon_amp.detach())

        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()
        lr_history.append(opt.param_groups[0]["lr"])
        history.append(sum(psnrs) / len(psnrs))
        if step in checkpoints:
            recon_by_checkpoint[step] = [r.cpu().numpy() for r in recon_planes]

    return slm_phase.detach(), history, recon_by_checkpoint, lr_history
