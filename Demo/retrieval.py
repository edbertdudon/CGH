"""
Phase retrieval algorithms for CGH.

multiplane_gs() implements a standard multi-plane Gerchberg-Saxton (GS)
iteration: at each step, propagate the current SLM field to every target
depth plane, replace the amplitude at each plane with the target amplitude
(keeping the computed phase), propagate each back to the SLM plane, average
the corrections, then re-enforce the phase-only constraint at the SLM
(amplitude=1, since a phase SLM cannot modulate amplitude).

This is one reasonable reading of the doc's "6 FFTs for iterative phase
retrieval" line -- but note GS is not the only or necessarily best choice.
Gradient-based methods (optimizing SLM phase directly via backprop through
the same propagation model, e.g. Adam on an amplitude-matching loss) are
now standard in CGH research and often reach a target quality in fewer
iterations. That path needs autodiff (torch/JAX) to be practical at real
resolution -- see sgd_stub() below for where to plug it in on a GPU
machine.
"""
import numpy as np
from propagation import angular_spectrum_propagate
from metrics import mean_psnr


def multiplane_gs(target_planes, depths_m, wavelength, dx, n_iters, seed=0):
    """
    Returns (slm_phase, history, recon_planes_by_iter)
      slm_phase: final phase-only SLM pattern
      history: list of mean PSNR across planes, one entry per iteration
      recon_planes_by_iter: dict{iteration_index: [recon amplitude per plane]}
        recorded at a few checkpoints for visualization
    """
    shape = target_planes[0].shape
    targets_amp = [np.sqrt(np.clip(t, 0, None)) for t in target_planes]

    rng = np.random.default_rng(seed)
    slm_phase = rng.uniform(-np.pi, np.pi, size=shape)
    slm_field = np.exp(1j * slm_phase)

    history = []
    checkpoints = sorted(set([1, max(1, n_iters // 2), n_iters]))
    recon_by_checkpoint = {}

    for it in range(1, n_iters + 1):
        correction = np.zeros(shape, dtype=complex)
        recon_planes = []
        for target_amp, z in zip(targets_amp, depths_m):
            field_at_plane = angular_spectrum_propagate(slm_field, wavelength, dx, z)
            recon_planes.append(np.abs(field_at_plane))
            phase_at_plane = np.angle(field_at_plane)
            constrained = target_amp * np.exp(1j * phase_at_plane)
            back = angular_spectrum_propagate(constrained, wavelength, dx, -z)
            correction += back

        slm_field_avg = correction / len(depths_m)
        slm_phase = np.angle(slm_field_avg)
        slm_field = np.exp(1j * slm_phase)  # phase-only constraint

        history.append(mean_psnr(recon_planes, target_planes))
        if it in checkpoints:
            recon_by_checkpoint[it] = recon_planes

    return slm_phase, history, recon_by_checkpoint


def sgd_stub():
    """
    Not implemented here (needs autodiff to be practical at real
    resolution). Sketch for a torch port on a GPU machine:

        slm_phase = torch.zeros(shape, requires_grad=True)
        opt = torch.optim.Adam([slm_phase], lr=0.02)
        for step in range(n_steps):
            field = torch.exp(1j * slm_phase)
            loss = 0
            for target_amp, z in zip(targets_amp, depths_m):
                recon = torch_angular_spectrum_propagate(field, wavelength, dx, z)
                loss = loss + ((recon.abs() - target_amp) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    Worth comparing directly against multiplane_gs() on the same targets:
    same quality threshold, count FFTs each needs to get there. That
    comparison alone could move the "14 FFTs/frame" number either way.
    """
    raise NotImplementedError("Port to torch on a GPU machine -- see docstring.")
