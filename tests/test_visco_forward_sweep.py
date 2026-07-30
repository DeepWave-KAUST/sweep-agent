"""Tests for ``run_visco_forward_sweep`` — the visco-acoustic solver-only path.

The defining behaviour: the two attenuation switches must actually change the
gather. phase-only and both must differ (the physics the equation demonstrates).
"""

from __future__ import annotations

import numpy as np
import pytest

from sweep_agent.tools import registry


def test_registered_with_the_attenuation_switches():
    assert "run_visco_forward_sweep" in registry.tools
    fields = registry.tools["run_visco_forward_sweep"].params_model.model_fields
    for required in ("vp_path", "dh", "dt", "q", "phase_shift", "amplitude_damping", "device"):
        assert required in fields, f"missing parameter: {required}"


def test_missing_solver_is_reported_as_data(monkeypatch):
    import sweep_agent.tools.visco_forward_sweep as vf

    monkeypatch.setattr(vf, "_require_sweep", lambda: (None, {"error": "no solver"}))
    out = vf.run_visco_forward_sweep.fn(
        vf.RunViscoForwardSweepParams(vp_path="/nonexistent.npy", dh=10.0, dt=1e-3, nt=100)
    )
    assert "error" in out


def test_selector_includes_run_visco_forward_sweep():
    from sweep_agent.tools.selection import select_tool_names

    assert "run_visco_forward_sweep" in select_tool_names("run an attenuating forward shot")


def _run(vf, tmp_path, vp_path, phase_shift, amplitude_damping):
    out = vf.run_visco_forward_sweep.fn(
        vf.RunViscoForwardSweepParams(
            vp_path=str(vp_path), dh=10.0, dt=1.0e-3, nt=300, fm=8.0, q=20.0,
            phase_shift=phase_shift, amplitude_damping=amplitude_damping,
            abcn=20, out_dir=str(tmp_path / f"{phase_shift}_{amplitude_damping}"),
            plot=False,
        )
    )
    assert "error" not in out, out
    return np.load(out["record_path"])


def test_toggles_change_the_gather(tmp_path):
    """phase-only and both must differ — the attenuation switches are live."""
    pytest.importorskip("sweep")
    pytest.importorskip("torch")

    import sweep_agent.tools.visco_forward_sweep as vf

    nz, nx = 80, 120
    vp = np.full((nz, nx), 2000.0, dtype=np.float32)
    vp[40:, :] = 2600.0
    vp_path = tmp_path / "vp.npy"
    np.save(vp_path, vp)

    phase_only = _run(vf, tmp_path, vp_path, phase_shift=True, amplitude_damping=False)
    both = _run(vf, tmp_path, vp_path, phase_shift=True, amplitude_damping=True)
    acoustic = _run(vf, tmp_path, vp_path, phase_shift=False, amplitude_damping=False)

    assert phase_only.shape == both.shape
    assert np.isfinite(both).all() and np.abs(both).max() > 0
    assert not np.allclose(phase_only, both), "amplitude_damping had no effect"
    assert not np.allclose(acoustic, phase_only), "phase_shift had no effect"
