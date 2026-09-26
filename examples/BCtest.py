"""
Analytical single-bar validation of k_elasticBoundaryTraction (the magnum-ported
free-surface traction condition) against a hand-checkable closed-form static
equilibrium, isolated from every other moving part in the elastodynamics port:

  - No magnetoelastic eigenstrain (no magnetization / B1 / B2 set), so sigMel
    and the face-eigenstrain jump offsets are exactly zero.
  - C12 = C44 = 0, so the only surviving interior force term is the pure
    diagonal d/dx( C11 * d(u_x)/dx ), and there are no C12/C44 cross terms to
    confound the boundary check.
  - No geometry cutouts, no material interfaces -- k_jumpCorrectGradient's
    interior-interface path is never exercised (Cii/Cshear are uniform
    everywhere), so any discrepancy can only come from k_elasticForceDirect's
    bulk term or k_elasticBoundaryTraction itself.
  - Only a single Cartesian direction (x) is ever non-interior, so this does
    NOT exercise the corner-averaging (multiple simultaneously non-interior
    directions) path in k_elasticBoundaryTraction -- that remains unverified
    by this test alone.

Physical setup: a thin bar along x, pulled outward by an equal and opposite
normal traction T on both end faces (no fixed/Dirichlet end -- mumax+ has no
displacement BC yet, only zero-traction-by-default boundaries with an optional
applied BoundaryTraction). At static equilibrium (f_total = 0 everywhere),
d/dx( C11 * d(u_x)/dx ) = 0 in the bulk, so d(u_x)/dx is a single constant
along the whole bar; matching sigma_xx = C11 * d(u_x)/dx = T at the loaded
faces fixes that constant directly:

    d(u_x)/dx = T / C11         (everywhere, exactly, in the continuum limit)

Because both ends are free (no anchor), the bar's average position is an
undetermined rigid-body mode -- only the SLOPE of u_x, and differences of
u_x between cells, are physically determined and checked here. u_y, u_z, and
the slopes of u_x in y/z should all remain exactly zero (or numerically
negligible) since nothing drives them.

Usage: this system is a genuine underdamped oscillator (rho*u'' + eta*u' -
elastic restoring force = 0), NOT a monotonic relaxation -- with the default
parameters below (C11=283e9, rho=8000, h=2e-9, eta=1e13), the per-cell natural
frequency is ~3e12 rad/s (~21 timesteps/cycle at dt=1e-13) and the momentum
damping time constant is tau=rho/eta~8e-10s (~8000 steps). A single-snapshot
convergence check aliases badly against this oscillation, so instead this test
runs a large fixed number of steps (default 150000, ~19 damping time
constants) and averages several trailing, phase-independent checkpoints, then
fits a line to u_x(x) and compares its slope to the analytic T/C11, checking
the fit residual to confirm u_x is linear (not just correctly sloped on
average) as the discretization predicts. Printed diagnostics report the
estimated frequency/damping timescale and flag if nsteps looks too short.
"""

import numpy as np
from mumaxplus import World, Grid, Ferromagnet


def run_bar_test(nx=40, h=2e-9, C11=283e9, T=5e6, eta=1e14,
                  dt=1e-13, nsteps=15000, check_every=500, tol_rtol=5e-2,
                  convergence_rtol=1e-3, verbose_convergence=True,
                  average_last_n_checkpoints=5):
    """
    nx       : number of cells along the bar
    h        : cellsize (isotropic; cross-section irrelevant for this 1D check)
    C11      : stiffness constant (Pa)
    T        : applied normal traction magnitude (Pa) -- +T pulls outward at
               +x face, -T pulls outward at -x face (see BoundaryTraction sign
               note below)
    eta      : viscous damping coefficient, large enough to reach equilibrium
               quickly without oscillation domination
    dt       : timestep
    nsteps   : max steps to run
    check_every : check for convergence (max |du_x/dt| proxy via successive
               snapshots) every this many steps, to avoid running longer than
               necessary and to report the convergence trend
    tol_rtol : relative tolerance on the fitted slope vs analytic T/C11

    Returns a dict with the fitted slope, analytic slope, relative error, and
    the linearity residual, and asserts the slope matches within tol_rtol.

    NOTE ON SIGN: this test does not assume a sign convention for
    BoundaryTraction.getSide going in. If the fitted slope comes out as
    -T/C11 instead of +T/C11, that is itself the direct, unambiguous
    confirmation that getSide's sign convention is the opposite of what
    k_elasticBoundaryTraction currently assumes -- see the assertion at the
    bottom, which checks the ABSOLUTE value of the slope first and reports
    the sign separately, so a sign flip is diagnosed rather than silently
    reported as "test failed, cause unknown."
    """
    world = World((h, h, h))
    grid = Grid((nx, 1, 1))
    magnet = Ferromagnet(world, grid)

    magnet.enable_elastodynamics = True
    magnet.C11 = C11
    magnet.C12 = 0
    magnet.C44 = 0
    magnet.rho = 8000.0
    magnet.eta = eta

    magnet.elastic_displacement = (0, 0, 0)

    magnet.boundary_traction.make_zero()
    magnet.boundary_traction.pos_x_side = (T, 0, 0)
    magnet.boundary_traction.neg_x_side = (-T, 0, 0)

    x = (np.arange(nx) + 0.5) * h  # cell-center coordinates
    analytic_slope = T / C11

    # Physical estimate (per-cell damped-oscillator approximation) of this
    # system's natural frequency and damping time constant, to sanity check
    # the run length / sampling interval BEFORE trusting any convergence
    # metric -- an underdamped system sampled at an interval close to (or an
    # integer multiple of) its own oscillation period will show large,
    # persistent-looking swings in a naive single-snapshot convergence check
    # even while the true envelope is decaying steadily. Rather than rely on
    # a single-snapshot rel_change criterion (which is exactly what aliases
    # against the oscillation), this test runs the FULL nsteps and then
    # averages several trailing checkpoints -- phase-random samples of a
    # decaying oscillation -- which converges toward the true equilibrium
    # slope regardless of instantaneous phase, provided nsteps is enough
    # damping time constants.
    rho_val = 8000.0  # must match magnet.rho set above
    omega0 = np.sqrt(C11 / (rho_val * h**2))
    tau_damp = rho_val / eta  # rho / eta
    period = 2 * np.pi / omega0
    n_tau_covered = nsteps * dt / tau_damp
    print(f"Estimated natural angular frequency omega0 ~ {omega0:.3e} rad/s "
          f"(period ~ {period:.3e} s, ~{period/dt:.1f} steps/cycle)")
    print(f"Estimated damping time constant tau = rho/eta ~ {tau_damp:.3e} s")
    print(f"Total run covers ~{n_tau_covered:.1f} damping time constants "
          f"({'looks sufficient, expect >99% decay' if n_tau_covered > 5 else 'MAY BE TOO SHORT -- increase nsteps'})")
    if check_every < 0.5 * period / dt:
        print(f"WARNING: check_every={check_every} samples faster than half the "
              f"oscillation period (~{period/dt:.0f} steps) -- checkpoints may "
              "still alias against the oscillation; this is fine as long as "
              "nsteps covers many tau_damp and the FINAL averaged slope is used, "
              "but individual checkpoint-to-checkpoint changes will look noisy.")

    history = []  # (step, slope_now)
    for step in range(0, nsteps, check_every):
        world.timesolver.run(check_every * dt)
        u = magnet.elastic_displacement.eval()  # shape (3, nz, ny, nx)
        ux = u[0, 0, 0, :]
        slope_now, _ = np.polyfit(x, ux, 1)
        history.append((step + check_every, slope_now, ux.copy()))
        if verbose_convergence:
            print(f"  step={step + check_every:7d}  slope={slope_now:.6e}  "
                  f"(target {analytic_slope:.6e}, ratio={slope_now/analytic_slope:.3%})")

    # Average the trailing checkpoints (phase-random samples of the -- by now,
    # hopefully mostly decayed -- oscillation) rather than trusting the very
    # last single snapshot, which is still just one phase sample.
    n_avg = min(average_last_n_checkpoints, len(history))
    trailing_slopes = [h_[1] for h_ in history[-n_avg:]]
    trailing_ux = np.mean([h_[2] for h_ in history[-n_avg:]], axis=0)
    slope = float(np.mean(trailing_slopes))
    slope_spread = float(np.std(trailing_slopes))
    ux = trailing_ux

    print(f"\nTrailing {n_avg} checkpoint slopes: {trailing_slopes}")
    print(f"Mean: {slope:.6e}  Std: {slope_spread:.3e} "
          f"({'still oscillating with non-negligible amplitude' if slope_spread > 0.05*abs(slope) else 'settled'})")

    converged_at = history[-1][0] if slope_spread < 0.05 * abs(slope) else None

    # linear fit of the trailing-averaged u_x(x) = slope*x + intercept
    # (this OVERWRITES the trailing-average `slope`/`slope_spread` above with
    # a fresh fit against the averaged profile -- both should closely agree
    # with each other if the trailing checkpoints truly are just phase-shifted
    # samples of the same decaying linear-in-x profile; a large disagreement
    # between the two would itself be worth investigating separately)
    slope, intercept = np.polyfit(x, ux, 1)
    fit = slope * x + intercept
    residual = ux - fit
    max_residual = np.max(np.abs(residual))
    ux_range = np.max(ux) - np.min(ux)
    relative_nonlinearity = max_residual / max(ux_range, 1e-30)

    result = {
        "converged_at_step": converged_at,
        "fitted_slope": slope,
        "analytic_slope": analytic_slope,
        "sign_matches": np.sign(slope) == np.sign(analytic_slope),
        "relative_slope_error": abs(abs(slope) - abs(analytic_slope)) / abs(analytic_slope),
        "relative_nonlinearity": relative_nonlinearity,
        "u_x": ux,
        "x": x,
    }

    print(f"Converged at step: {converged_at}")
    print(f"Fitted slope:   {slope:.6e}")
    print(f"Analytic slope: {analytic_slope:.6e} (T/C11)")
    print(f"Sign match: {result['sign_matches']}")
    print(f"Relative slope error (magnitude only): {result['relative_slope_error']:.3%}")
    print(f"Relative nonlinearity (fit residual / u_x range): {relative_nonlinearity:.3%}")

    if not result["sign_matches"]:
        print("\n*** SIGN MISMATCH: fitted slope has opposite sign from T/C11. ***")
        print("*** This means BoundaryTraction.getSide's sign convention is  ***")
        print("*** opposite to what k_elasticBoundaryTraction assumes.       ***")

    assert result["sign_matches"], (
        f"Sign mismatch: fitted slope {slope:.3e} vs analytic {analytic_slope:.3e}. "
        "getSide's sign convention is likely inverted relative to what "
        "k_elasticBoundaryTraction assumes -- flip the sign in the kernel."
    )
    assert result["relative_slope_error"] < tol_rtol, (
        f"Slope magnitude off by {result['relative_slope_error']:.3%} "
        f"(tolerance {tol_rtol:.3%}): fitted {slope:.6e}, analytic {analytic_slope:.6e}"
    )
    assert relative_nonlinearity < 0.05, (
        f"u_x(x) is not linear as expected (residual {relative_nonlinearity:.3%} "
        "of total range) -- suggests the bulk interior operator "
        "(k_elasticForceDirect) or boundary kernel has an error beyond just "
        "the boundary slope, e.g. a jump/discontinuity being introduced "
        "somewhere along the bar."
    )

    return result


if __name__ == "__main__":
    run_bar_test()
