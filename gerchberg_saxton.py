"""
gerchberg_saxton.py

Classic iterative phase retrieval, built entirely on top of the already
validated AngularSpectrumPropagation layer. No neural network here -- this
is the "does the physics work on a real target" sanity check from the
roadmap.

In the arrow language from the derivation: force the amplitude to match
the target at the image plane (keep whatever phase came out of the last
propagation), force the amplitude back to 1 at the SLM plane (phase-only
hardware can't do anything else), and keep bouncing between the two until
the guess settles into something self-consistent.
"""

import torch


def gerchberg_saxton(target_amplitude, propagator, z, n_iters=200, seed=0):
    """
    Solve for a phase-only SLM pattern that reconstructs target_amplitude
    at distance z, using the given AngularSpectrumPropagation layer.

    Parameters
    ----------
    target_amplitude : torch.Tensor, real, shape (ny, nx)
        Desired brightness pattern at the image plane. Only its *shape*
        matters, not its absolute scale -- it's rescaled internally to
        match the SLM plane's total energy.
    propagator : AngularSpectrumPropagation
        The propagation layer, already validated against the lens test.
    z : float
        Distance from the SLM to the target image plane.
    n_iters : int
        Number of forward/backward round trips.
    seed : int
        Random seed for the initial phase guess.

    Returns
    -------
    slm_phase : torch.Tensor, real, shape (ny, nx)
        The recovered phase-only pattern, in radians.
    errors : list[float]
        Reconstruction error (normalized RMSE) at every iteration -- plot
        this to confirm the loop is actually converging, not stuck.
    """
    device = target_amplitude.device
    ny, nx = target_amplitude.shape

    # Scale the target so its total energy matches the SLM plane's energy
    # (amplitude 1 everywhere -> energy = ny * nx). Keeps both planes on
    # a comparable scale; only the target's shape actually matters.
    slm_energy = (ny * nx) ** 0.5
    target_scale = slm_energy / torch.linalg.vector_norm(target_amplitude)
    target_scaled = target_amplitude * target_scale
    target_unit = target_amplitude / torch.linalg.vector_norm(target_amplitude)

    generator = torch.Generator(device=device).manual_seed(seed)
    random_phase = torch.rand(ny, nx, device=device, generator=generator) * 2 * torch.pi
    slm_field = torch.exp(1j * random_phase)

    errors = []
    for _ in range(n_iters):
        # forward: SLM plane -> image plane
        image_field = propagator(slm_field, z)

        # how close is the current reconstruction, ignoring absolute scale?
        image_amp = image_field.abs()
        recon_unit = image_amp / torch.linalg.vector_norm(image_amp)
        err = torch.sqrt(torch.mean((recon_unit - target_unit) ** 2))
        errors.append(err.item())

        # force amplitude to the target, keep the phase that came out
        image_phase = torch.angle(image_field)
        corrected_image = target_scaled * torch.exp(1j * image_phase)

        # backward: image plane -> SLM plane (negative z undoes the propagation)
        slm_field_back = propagator(corrected_image, -z)

        # force amplitude back to 1 (phase-only hardware), keep the phase
        slm_phase = torch.angle(slm_field_back)
        slm_field = torch.exp(1j * slm_phase)

    return slm_phase, errors
