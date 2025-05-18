import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import quik_bind as quikpy


def init_quik_robot(eff_a: np.ndarray, eff_alpha: np.ndarray, eff_d: np.ndarray) -> None:
    """Initialize QuIK robot with DH parameters."""
    # Create DH parameter matrix for QuIK (a, alpha, d, theta_offset)
    dh = np.zeros((6, 4))
    dh[:, 0] = eff_a
    dh[:, 1] = eff_alpha
    dh[:, 2] = eff_d
    # All joints are revolute
    link_types = np.zeros(6, dtype=bool)  # False = revolute, True = prismatic
    quikpy.init_robot(dh, link_types)


def dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Legacy DH matrix calculation."""
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
    use_quik: bool = True,
) -> np.ndarray:
    """
    Compute forward kinematics to the flange frame.
    
    Args:
        eff_a: Effective link lengths
        eff_alpha: Effective twist angles
        eff_d: Effective link offsets
        _joint_dirs_unused: Joint directions (unused)
        dh_thetas: Joint angles in radians
        use_quik: Whether to use QuIK's FK solver (default: True)
    
    Returns:
        4x4 homogeneous transformation matrix from base to flange
    """
    if use_quik:
        try:
            # Initialize robot if not already initialized
            init_quik_robot(eff_a, eff_alpha, eff_d)
            return quikpy.fkn(dh_thetas)
        except Exception as e:
            print(f"Warning: QuIK FK failed ({e}), falling back to Python implementation")
            use_quik = False
    
    if not use_quik:
        # Legacy Python implementation
        T = np.eye(4)
        for i in range(len(eff_a)):
            T = T @ dh_matrix(eff_a[i], eff_alpha[i], eff_d[i], dh_thetas[i])
        return T


def tcp_transform(tcp_pose_list: np.ndarray) -> np.ndarray:
    """Transform from flange to TCP frame."""
    T_tcp = np.eye(4)
    T_tcp[0:3, 3] = tcp_pose_list[0:3]
    rotvec = tcp_pose_list[3:6]
    if np.linalg.norm(rotvec) > 1e-9:
        T_tcp[0:3, 0:3] = R_scipy.from_rotvec(rotvec).as_matrix()
    return T_tcp
