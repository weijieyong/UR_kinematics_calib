import numpy as np
from scipy.spatial.transform import Rotation as R_scipy


def dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )


def fk_to_flange(
    eff_a: np.ndarray,
    eff_alpha: np.ndarray,
    eff_d: np.ndarray,
    _joint_dirs_unused: np.ndarray,
    dh_thetas: np.ndarray,
) -> np.ndarray:
    T = np.eye(4)
    for i in range(len(eff_a)):
        T = T @ dh_matrix(eff_a[i], eff_alpha[i], eff_d[i], dh_thetas[i])
    return T


def tcp_transform(tcp_pose_list: np.ndarray) -> np.ndarray:
    T_tcp = np.eye(4)
    T_tcp[0:3, 3] = tcp_pose_list[0:3]
    rotvec = tcp_pose_list[3:6]
    if np.linalg.norm(rotvec) > 1e-9:
        T_tcp[0:3, 0:3] = R_scipy.from_rotvec(rotvec).as_matrix()
    return T_tcp
