from ur_kinematics_calib.fk import fk_to_flange
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares
import numpy as np
import quik_bind as quikpy

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

def ik_quik(eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, q_init=None):
    """
    QuIK-based IK solver using Python bindings.
    Much faster and more robust than the numerical solver.
    
    Returns:
        Tuple: (solution, extra_data)
            - solution: joint angles (without dt offset)
            - extra_data: (error, iterations, reason)
    """
    if q_init is None:
        q_init = np.zeros(6)
    
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
    
    # Adjust target for the TCP offset
    T_target_fl = T_target @ np.linalg.inv(T_fl_tcp)
    
    # Solve IK
    q_sol, e_sol, iterations, reason = quikpy.ik(T_target_fl, q_init)
    
    # Return solution and additional info
    return q_sol, (e_sol, iterations, reason)