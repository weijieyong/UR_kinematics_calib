#!/usr/bin/env python3
"""UR Dry-Parse: Calibration & FK Tool."""

import os
import sys
import ast
import configparser
import argparse
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
    ), np.array(parse_conf_list(tool.get("tcp_pose", "[0,0,0,0,0,0]")), dtype=np.float64)


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


def fk_to_flange(eff_a, eff_alpha, eff_d, _joint_dirs_unused, dh_thetas) -> np.ndarray:
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


def main():
    parser = argparse.ArgumentParser(description="UR Dry-Parse: Calibration & FK")
    parser.add_argument("--fk", type=str, help="Joints(deg), for FK")
    parser.add_argument(
        "--compare",
        type=str,
        help="Joints(deg),MeasuredTCP[Xmm,Ymm,Zmm,Rx,Ry,Rz], for Pose Comparison",
    )
    args = parser.parse_args()

    s_dir = os.path.dirname(os.path.abspath(__file__))
    c_path = os.path.join(s_dir, "configs", "calibration.conf")
    u_path = os.path.join(s_dir, "configs", "urcontrol.conf.UR5")

    if not (os.path.exists(c_path) and os.path.exists(u_path)):
        print("Error: Config file(s) not found in ./configs/")
        sys.exit(1)

    dt, da, dd, dalpha = load_calibration(c_path)
    (a0, d0, alpha0, q_home0, j_dir), tcp_conf = load_urcontrol_config(u_path)

    eff_a, eff_d, eff_alpha = a0 + da, d0 + dd, alpha0 + dalpha

    print("Effective D-H (a, d in m; alpha in rad):")
    print(f"  a_eff:     {eff_a.round(6).tolist()}")
    print(f"  d_eff:     {eff_d.round(6).tolist()}")
    print(f"  alpha_eff: {eff_alpha.round(6).tolist()}")

    T_fl_tcp = tcp_transform(tcp_conf)
    # print(f"\nTCP Transform (from flange):\n{T_fl_tcp.round(4)}") # Optional: if needed

    q_deg_in, meas_tcp_m_rad = None, None

    if args.compare:
        try:
            v = [float(x) for x in args.compare.split(",")]
            if len(v) != 12:
                raise ValueError("Expected 12 values for --compare")
            q_deg_in, meas_tcp_m_rad = np.array(v[:6]), np.array(v[6:])
        except ValueError as e:
            print(f"Error (--compare): {e}")
            sys.exit(1)
    elif args.fk:
        try:
            q_deg_in = np.array([float(x) for x in args.fk.split(",")])
            if len(q_deg_in) != 6:
                raise ValueError("Expected 6 values for --fk")
        except ValueError as e:
            print(f"Error (--fk): {e}")
            sys.exit(1)

    dh_thetas_fk = np.zeros(6)
    if q_deg_in is not None:
        print(f"\nInput Joints (deg): {q_deg_in.tolist()}")
        dh_thetas_fk = np.deg2rad(q_deg_in) + dt
    else:
        print("\nNo joints given, using effective 'home' config.")
        dh_thetas_fk = q_home0 + dt  # User's q_home_eff
    # print(f"D-H Thetas for FK (rad): {dh_thetas_fk.round(6).tolist()}") # Optional

    T_base_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas_fk)
    T_base_tcp = T_base_fl @ T_fl_tcp

    pos_calc_m = T_base_tcp[0:3, 3]
    rotvec_calc_rad = R_scipy.from_matrix(T_base_tcp[0:3, 0:3]).as_rotvec()

    print("\nCalculated TCP Pose:")
    print(f"  Pos (mm): {(pos_calc_m * 1000.0).round(4).tolist()}")
    print(f"  RotVec (rad): {rotvec_calc_rad.round(6).tolist()}")

    if meas_tcp_m_rad is not None:
        compare_poses(T_base_tcp, meas_tcp_m_rad)
    elif not (args.compare or args.fk):
        try:
            usr_in = input(
                "Enter Measured TCP [Xm,Ym,Zm,Rx,Ry,Rz] or Enter to skip: "
            ).strip()
            if usr_in:
                v = [float(x) for x in usr_in.split(",")]
                if len(v) != 6:
                    raise ValueError("Expected 6 values")
                compare_poses(T_base_tcp, np.array(v))
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
