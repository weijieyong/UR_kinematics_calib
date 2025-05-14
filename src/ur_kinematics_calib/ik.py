from ur_kinematics_calib.fk import fk_to_flange
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares
import numpy as np

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