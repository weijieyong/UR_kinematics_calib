import ast
import configparser
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy


def parse_conf_list(value: str):
    return ast.literal_eval(value.split("#", 1)[0].strip())


def load_calibration(calib_path: str):
    config = configparser.ConfigParser(strict=False)
    with open(calib_path, "r") as f:
        config.read_file(f)
    mount = config["mounting"]
    if int(mount.get("calibration_status", "0").split()[0]) != 2:
        print("Warning: Calibration status not 2 (Linearised).")
    return (
        np.array(parse_conf_list(mount.get(k, "[0,0,0,0,0,0]")), dtype=np.float64)
        for k in ["delta_theta", "delta_a", "delta_d", "delta_alpha"]
    )


def load_urcontrol_config(ur_path: str):
    config = configparser.ConfigParser(strict=False)
    with open(ur_path, "r") as f:
        config.read_file(f)
    dh = config["DH"]
    tool = config["Tool"]
    return (
        np.array(parse_conf_list(dh.get(k, "[0,0,0,0,0,0]")), dtype=np.float64)
        for k in ["a", "d", "alpha", "q_home_offset", "joint_direction"]
    ), np.array(
        parse_conf_list(tool.get("tcp_pose", "[0,0,0,0,0,0]")), dtype=np.float64
    )


def compare_poses(T_calc_tcp: np.ndarray, measured_tcp_m_rad: np.ndarray):
    pos_calc_mm = T_calc_tcp[0:3, 3] * 1000.0
    R_calc = T_calc_tcp[0:3, 0:3]

    pos_meas_mm = measured_tcp_m_rad[0:3]
    R_meas = R_scipy.from_rotvec(measured_tcp_m_rad[3:6]).as_matrix()

    pos_err_mm = pos_calc_mm - pos_meas_mm
    pos_err_norm_mm = np.linalg.norm(pos_err_mm)

    rot_err_angle_rad = 0.0
    try:
        R_err_mat = R_calc @ R_meas.T  # R_meas.T is R_meas_inv for rotation matrices
        rot_err_angle_rad = R_scipy.from_matrix(R_err_mat).magnitude()
    except Exception:
        rot_err_angle_rad = float("nan")

    print("\n--- Pose Comparison (Calculated vs. Measured) ---")
    print(f"  Calc Pos (mm): {pos_calc_mm.round(4).tolist()}")
    print(f"  Meas Pos (mm): {pos_meas_mm.round(4).tolist()}")
    print(
        f"  Pos Error (mm): {pos_err_mm.round(4).tolist()}, Norm (mm): {pos_err_norm_mm:.4f}"
    )
    print(f"  Rot Error (deg): {np.rad2deg(rot_err_angle_rad):.4f}")
