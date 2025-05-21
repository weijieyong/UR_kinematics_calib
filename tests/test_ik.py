import numpy as np
import pytest
from ur_kinematics_calib.ik import wrap_angle, select_branch, ik_quik, analytic_ik_nominal
from ur_kinematics_calib.fk import fk_to_flange

def test_wrap_angle():
    diff = np.array([np.pi + 0.1])
    wrapped = wrap_angle(diff)
    expected = np.array([-np.pi + 0.1])
    assert np.allclose(wrapped, expected)

@pytest.mark.parametrize(
    "input_val, expected",
    [
        (np.array([0]), np.array([0])),
        (np.array([2*np.pi]), np.array([0])),
        (np.array([-2*np.pi]), np.array([0])),
        (np.array([3*np.pi/2]), np.array([-np.pi/2])),
    ],
)
def test_wrap_angle_param(input_val, expected):
    assert np.allclose(wrap_angle(input_val), expected)

def test_select_branch_simple():
    q_branches = [np.zeros(6), np.array([np.pi] + [0] * 5)]
    q_ref = np.zeros(6)
    sel = select_branch(q_branches, q_ref)
    assert np.allclose(sel, np.zeros(6))
    q_ref2 = np.array([np.pi, 0, 0, 0, 0, 0])
    sel2 = select_branch(q_branches, q_ref2)
    assert np.allclose(sel2, np.array([np.pi] + [0] * 5))

def test_select_branch_empty():
    assert np.allclose(select_branch([], np.array([1,2,3,4,5,6])), np.zeros(6))

def test_select_branch_penalty():
    branch1 = np.zeros(6)
    branch2 = np.ones(6) * (3*np.pi)
    q_ref = np.zeros(6)
    sel = select_branch([branch1, branch2], q_ref)
    assert np.allclose(sel, branch1)

def test_analytic_ik_nominal_returns_list():
    sols = analytic_ik_nominal(np.eye(4))
    assert isinstance(sols, list)
    if sols:
        for sol in sols:
            assert len(sol) == 6

def test_ik_quik_with_monkeypatch(monkeypatch):
    import ur_kinematics_calib.ik as ik_mod
    monkeypatch.setattr(ik_mod.quikpy, 'init_robot', lambda dh, link_types: None)
    dummy_q = np.array([1, 2, 3, 4, 5, 6])
    def fake_ik(T_target_fl, q_init):
        return dummy_q, 0.0, 10, 'ok'
    monkeypatch.setattr(ik_mod.quikpy, 'ik', fake_ik)
    eff_a = np.zeros(6)
    eff_alpha = np.zeros(6)
    eff_d = np.zeros(6)
    j_dir = np.zeros(6)
    dt = np.zeros(6)
    T_fl_tcp = np.eye(4)
    T_target = np.eye(4)
    q_sol, extra = ik_mod.ik_quik(eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, q_init=None, use_analytic_seed=False)
    assert np.allclose(q_sol, dummy_q)
    # Updated to expect 4 values in extra including is_reachable flag
    e_sol, iterations, reason, is_reachable = extra
    assert e_sol == 0.0
    assert iterations == 10
    assert reason == 'ok'
    assert is_reachable is True  # Should be reachable with zero error

def test_ik_quik_uses_current_joints(monkeypatch):
    import ur_kinematics_calib.ik as ik_mod
    calls = {}
    monkeypatch.setattr(ik_mod.quikpy, 'init_robot', lambda dh, link_types: None)
    branches = [np.arange(6)]
    monkeypatch.setattr(ik_mod, 'analytic_ik_nominal', lambda T: branches)
    def fake_select(q_branches, q_ref):
        calls['q_branches'] = q_branches
        calls['q_ref'] = q_ref.copy()
        return q_ref + 1.0
    monkeypatch.setattr(ik_mod, 'select_branch', fake_select)
    monkeypatch.setattr(ik_mod.quikpy, 'ik', lambda T_target_fl, q_init: (q_init, 0.0, 1, 'ok'))
    eff_a = eff_alpha = eff_d = j_dir = np.zeros(6)
    dt = np.zeros(6)
    T_fl_tcp = T_target = np.eye(4)
    current = np.arange(6)
    q_sol, extra = ik_mod.ik_quik(
        eff_a, eff_alpha, eff_d, j_dir, dt,
        T_fl_tcp, T_target,
        q_init=None,
        use_analytic_seed=True,
        current_joints=current
    )
    assert np.allclose(calls['q_branches'], branches)
    assert np.allclose(calls['q_ref'], current)
    assert np.allclose(q_sol, current + 1.0)
    # No need to check all elements, just make sure structure is correct
    _, _, _, is_reachable = extra
    assert is_reachable is True

def test_ik_quik_uses_j_dir_when_no_current(monkeypatch):
    import ur_kinematics_calib.ik as ik_mod
    calls = {}
    monkeypatch.setattr(ik_mod.quikpy, 'init_robot', lambda dh, link_types: None)
    branches = [np.arange(6)]
    monkeypatch.setattr(ik_mod, 'analytic_ik_nominal', lambda T: branches)
    def fake_select(q_branches, q_ref):
        calls['q_ref'] = q_ref.copy()
        return np.zeros(6)
    monkeypatch.setattr(ik_mod, 'select_branch', fake_select)
    monkeypatch.setattr(ik_mod.quikpy, 'ik', lambda T_target_fl, q_init: (q_init, 0.0, 1, 'ok'))
    eff_a = eff_alpha = eff_d = j_dir = np.ones(6)
    dt = np.zeros(6)
    T_fl_tcp = T_target = np.eye(4)
    q_sol, extra = ik_mod.ik_quik(
        eff_a, eff_alpha, eff_d, j_dir, dt,
        T_fl_tcp, T_target,
        q_init=None,
        use_analytic_seed=True
    )
    assert np.allclose(calls['q_ref'], j_dir)
    # Make sure extra has the right structure
    _, _, _, is_reachable = extra
    assert is_reachable is True

def test_fk_ik_quik_roundtrip_identity(monkeypatch):
    import quik_bind as quikpy
    from ur_kinematics_calib.ik import ik_quik
    from ur_kinematics_calib.fk import fk_to_flange
    # Trivial robot parameters yielding identity FK
    eff_a = np.zeros(6)
    eff_alpha = np.zeros(6)
    eff_d = np.zeros(6)
    j_dir = np.zeros(6)
    dt = np.zeros(6)
    T_fl_tcp = np.eye(4)
    T_target = np.eye(4)
    # Monkeypatch QuIK to use q_init as solution
    monkeypatch.setattr(quikpy, 'init_robot', lambda dh, lt: None)
    monkeypatch.setattr(quikpy, 'ik', lambda T_target_fl, q_init: (q_init, 0.0, 0, 'ok'))
    # Provide initial guess equal to zero angles
    q_init = np.zeros(6)
    q_sol, _ = ik_quik(
        eff_a, eff_alpha, eff_d, j_dir, dt,
        T_fl_tcp, T_target,
        q_init=q_init
    )
    # FK of the IK solution (using Python FK) should reproduce the target pose
    T_sol = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, q_sol, use_quik=False)
    assert np.allclose(T_sol, T_target)

def test_ik_quik_detects_unreachable(monkeypatch):
    import ur_kinematics_calib.ik as ik_mod
    monkeypatch.setattr(ik_mod.quikpy, 'init_robot', lambda dh, link_types: None)
    
    # Mock a failed IK solution with high error and BREAKREASON_GRAD_FAILS
    def fake_ik_unreachable(T_target_fl, q_init):
        return np.zeros(6), 0.1, 100, "BREAKREASON_GRAD_FAILS"
    
    monkeypatch.setattr(ik_mod.quikpy, 'ik', fake_ik_unreachable)
    
    eff_a = eff_alpha = eff_d = j_dir = dt = np.zeros(6)
    T_fl_tcp = T_target = np.eye(4)
    
    q_sol, extra = ik_mod.ik_quik(
        eff_a, eff_alpha, eff_d, j_dir, dt,
        T_fl_tcp, T_target,
        q_init=np.zeros(6),
        error_threshold=0.01  # Set a lower threshold to ensure the test fails
    )
    
    _, _, reason, is_reachable = extra
    assert not is_reachable
    assert reason == "BREAKREASON_GRAD_FAILS"
