from ur_kinematics_calib.fk import fk_to_flange
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares
import numpy as np
import quik_bind as quikpy
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(filename)s:%(lineno)d] %(message)s'
)

# Analytic IK for UR robot, ported from https://github.com/ros-industrial/universal_robot/tree/noetic-devel/ur_kinematics 
# Only supports UR5e params by default; for other models, update the DH constants accordingly.
def analytic_ik_nominal(T):
    """
    Analytic IK for UR robot. Returns all valid solutions (up to 8).
    T: 4x4 numpy array (flange pose in base frame)
    Returns: list of 6-element numpy arrays (joint angles in radians)
    """
    # UR5 DH constants (SI units, meters)
    d1 = 0.1625
    a2 = -0.42500
    a3 = -0.3922
    d4 = 0.1333
    d5 = 0.0997
    d6 = 0.0996
    ZERO_THRESH = 1e-8
    PI = np.pi
    
    # Unpack T (C++ code uses column-major, numpy is row-major)
    T = np.asarray(T)
    T = T.copy()
    
    T00, T01, T02, T03, T10, T11, T12, T13, T20, T21, T22, T23 = T[0,0], T[0,1], T[0,2], T[0,3], T[1,0], T[1,1], T[1,2], T[1,3], T[2,0], T[2,1], T[2,2], T[2,3]
    
    # DEBUG: Print unpacked T and computed A, B, R
    logging.debug(f"T00={T00:.4f}, T01={T01:.4f}, T02={T02:.4f}, T03={T03:.4f}")
    logging.debug(f"T10={T10:.4f}, T11={T11:.4f}, T12={T12:.4f}, T13={T13:.4f}")
    logging.debug(f"T20={T20:.4f}, T21={T21:.4f}, T22={T22:.4f}, T23={T23:.4f}")
    
    q_sols = []
    # Shoulder rotate joint (q1)
    A = d6*T12 - T13
    B = d6*T02 - T03
    R = A*A + B*B
    q1 = [None, None]
    if abs(A) < ZERO_THRESH:
        div = -SIGN(d4)*SIGN(B) if abs(abs(d4) - abs(B)) < ZERO_THRESH else -d4/B
        arcsin = np.arcsin(div)
        arcsin = 0.0 if abs(arcsin) < ZERO_THRESH else arcsin
        q1[0] = arcsin + 2*PI if arcsin < 0.0 else arcsin
        q1[1] = PI - arcsin
    elif abs(B) < ZERO_THRESH:
        div = SIGN(d4)*SIGN(A) if abs(abs(d4) - abs(A)) < ZERO_THRESH else d4/A
        arccos = np.arccos(div)
        q1[0] = arccos
        q1[1] = 2*PI - arccos
    elif d4*d4 > R:
        return []
    else:
        arccos = np.arccos(d4 / np.sqrt(R))
        arctan = np.arctan2(-B, A)
        pos = arccos + arctan
        neg = -arccos + arctan
        pos = 0.0 if abs(pos) < ZERO_THRESH else pos
        neg = 0.0 if abs(neg) < ZERO_THRESH else neg
        q1[0] = pos if pos >= 0.0 else 2*PI + pos
        q1[1] = neg if neg >= 0.0 else 2*PI + neg
    # Wrist 2 joint (q5)
    q5 = [[None, None], [None, None]]
    for i in range(2):
        numer = (T03*np.sin(q1[i]) - T13*np.cos(q1[i]) - d4)
        div = SIGN(numer) * SIGN(d6) if abs(abs(numer) - abs(d6)) < ZERO_THRESH else numer / d6
        arccos = np.arccos(div)
        q5[i][0] = arccos
        q5[i][1] = 2*PI - arccos
    # After q1
    logging.debug(f"q1: {q1}")
    # After q5
    logging.debug(f"q5: {q5}")
    # Main loop for all branches
    for i in range(2):
        for j in range(2):
            c1 = np.cos(q1[i])
            s1 = np.sin(q1[i])
            c5 = np.cos(q5[i][j])
            s5 = np.sin(q5[i][j])
            # Wrist 3 joint (q6)
            if abs(s5) < ZERO_THRESH:
                q6 = 0.0  # Arbitrary, since q6 is indeterminate
            else:
                q6 = np.arctan2(SIGN(s5)*-(T01*s1 - T11*c1), SIGN(s5)*(T00*s1 - T10*c1))
                q6 = 0.0 if abs(q6) < ZERO_THRESH else q6
                if q6 < 0.0:
                    q6 += 2*PI
            c6 = np.cos(q6)
            s6 = np.sin(q6)
            x04x = -s5*(T02*c1 + T12*s1) - c5*(s6*(T01*c1 + T11*s1) - c6*(T00*c1 + T10*s1))
            x04y = c5*(T20*c6 - T21*s6) - T22*s5
            p13x = d5*(s6*(T00*c1 + T10*s1) + c6*(T01*c1 + T11*s1)) - d6*(T02*c1 + T12*s1) + T03*c1 + T13*s1
            p13y = T23 - d1 - d6*T22 + d5*(T21*c6 + T20*s6)
            c3 = (p13x*p13x + p13y*p13y - a2*a2 - a3*a3) / (2.0*a2*a3)
            if abs(abs(c3) - 1.0) < ZERO_THRESH:
                c3 = SIGN(c3)
            elif abs(c3) > 1.0:
                continue  # No solution
            arccos = np.arccos(c3)
            q3 = [arccos, 2*PI - arccos]
            denom = a2*a2 + a3*a3 + 2*a2*a3*c3
            s3 = np.sin(arccos)
            A = (a2 + a3*c3)
            B = a3*s3
            q2 = [np.arctan2((A*p13y - B*p13x) / denom, (A*p13x + B*p13y) / denom),
                  np.arctan2((A*p13y + B*p13x) / denom, (A*p13x - B*p13y) / denom)]
            c23_0 = np.cos(q2[0]+q3[0])
            s23_0 = np.sin(q2[0]+q3[0])
            c23_1 = np.cos(q2[1]+q3[1])
            s23_1 = np.sin(q2[1]+q3[1])
            q4 = [np.arctan2(c23_0*x04y - s23_0*x04x, x04x*c23_0 + x04y*s23_0),
                  np.arctan2(c23_1*x04y - s23_1*x04x, x04x*c23_1 + x04y*s23_1)]
            # In main loop, print each solution
            for k in range(2):
                q2k = q2[k]
                q3k = q3[k]
                q4k = q4[k]
                q2k = 0.0 if abs(q2k) < ZERO_THRESH else (q2k + 2*PI if q2k < 0.0 else q2k)
                q4k = 0.0 if abs(q4k) < ZERO_THRESH else (q4k + 2*PI if q4k < 0.0 else q4k)
                sol = np.array([
                    q1[i], q2k, q3k, q4k, q5[i][j], q6
                ])
                logging.debug(f"Branch: i={i}, j={j}, k={k}, q1={q1[i]:.4f}, q2={q2k:.4f}, q3={q3k:.4f}, q4={q4k:.4f}, q5={q5[i][j]:.4f}, q6={q6:.4f}")
                q_sols.append(sol)
    return q_sols


JOINT_LIMS = np.array([
    [-2*np.pi,  2*np.pi],   # q0 … q5
    [-2*np.pi,  2*np.pi],
    [-2*np.pi,  2*np.pi],
    [-2*np.pi,  2*np.pi],
    [-2*np.pi,  2*np.pi],
    [-2*np.pi,  2*np.pi],
])

def wrap_angle(diff):
    """Shortest angular distance, element-wise, in (–π, π]."""
    return (diff + np.pi) % (2*np.pi) - np.pi

def select_branch(q_branches, q_ref, joint_limits=JOINT_LIMS):
    """
    Choose the analytic branch closest to q_ref **modulo 2π** and
    with a penalty for violating limits.
    """
    if not q_branches:
        return np.zeros(6)

    q_ref = np.asarray(q_ref, dtype=float)
    best_score, best_q = np.inf, None

    for q in q_branches:
        q = np.asarray(q, dtype=float)

        # Distance in joint space with 2π wrapping
        d = wrap_angle(q - q_ref)
        score = np.dot(d, d)

        # Add a big penalty if the branch is out of limits
        if joint_limits is not None:
            out = (q < joint_limits[:, 0]) | (q > joint_limits[:, 1])
            #print only if out contains any True
            if np.any(out):
                print(f"Branch out of limits: {q}, out={out}")
            score += 1e3 * np.count_nonzero(out)

        if score < best_score:
            best_score, best_q = score, q
    
    return best_q

def ik_numerical(eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, q_init=None):
    """
    Numerical IK solver using least-squares.
    note that this implementation may cause IK to fail to converge/stuck in local minima if the initial guess is not close to the solution.
    """
    if q_init is None:
        q_init = np.zeros(6)

    def error_func(q):
        dh_thetas = q + dt
        T_base_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas)
        T_base_tcp = T_base_fl @ T_fl_tcp
        pos_cur = T_base_tcp[0:3, 3]
        R_cur = T_base_tcp[0:3, 0:3]
        pos_tgt = T_target[0:3, 3]
        R_tgt = T_target[0:3, 0:3]
        pos_err = pos_cur - pos_tgt
        R_err = R_cur @ R_tgt.T
        rot_err = R_scipy.from_matrix(R_err).as_rotvec()
        return np.concatenate([pos_err, rot_err])

    res = least_squares(error_func, q_init, xtol=1e-6, ftol=1e-6)
    return res.x, res

def ik_quik(
    eff_a, eff_alpha, eff_d, j_dir, dt,
    T_fl_tcp, T_target,
    q_init: np.ndarray | None = None,
    *,
    use_analytic_seed: bool = True,
) -> tuple[np.ndarray, tuple[float, int, str]]:
    """
    QuIK-based IK solver using Python bindings.
    Much faster and more robust than the numerical solver.
    Optionally uses analytic IK for initial guess.
    
    Returns:
        Tuple: (solution, extra_data)
            - solution: joint angles (without dt offset)
            - extra_data: (error, iterations, reason)
    """
    if q_init is None:
        # Adjust target for the TCP offset (needed for analytic IK too)
        T_target_fl = T_target @ np.linalg.inv(T_fl_tcp)
        if use_analytic_seed:
            # Use analytic IK for initial guess
            # analytic_ik_nominal should return a list/array of possible solutions
            q_branches = analytic_ik_nominal(T_target_fl)
            # select_branch should pick the best branch based on j_dir or other heuristics
            q_init = select_branch(q_branches, j_dir)
        else:
            q_init = np.zeros(6)
    else:
        # Adjust target for the TCP offset
        T_target_fl = T_target @ np.linalg.inv(T_fl_tcp)

    # Create DH parameters in the format QuIK expects [a, alpha, d, theta]
    dh_params = np.zeros((6, 4))
    for i in range(6):
        dh_params[i, 0] = eff_a[i]
        dh_params[i, 1] = eff_alpha[i]
        dh_params[i, 2] = eff_d[i]
        dh_params[i, 3] = dt[i]  # Initial offset (will be overwritten)

    # All revolute joints for UR robot
    link_types = np.array([False] * 6, dtype=bool)

    # Initialize the robot with DH parameters
    quikpy.init_robot(dh_params, link_types)

    # Solve IK
    q_sol, e_sol, iterations, reason = quikpy.ik(T_target_fl, q_init)

    # Return solution and additional info
    return q_sol, (e_sol, iterations, reason)

def SIGN(x):
    return int(x > 0) - int(x < 0)