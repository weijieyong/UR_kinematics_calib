#!/usr/bin/env python3
"""UR Dry-Parse: Calibration & FK Tool."""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from ur_kinematics_calib.util import (
    load_calibration,
    load_urcontrol_config,
    compare_poses,
)
from ur_kinematics_calib.fk import fk_to_flange, tcp_transform


def main():
    parser = argparse.ArgumentParser(description="UR Dry-Parse: Calibration & FK")
    parser.add_argument("--fk", type=str, help="Joints(deg), for FK")
    parser.add_argument(
        "--compare",
        type=str,
        help="Joints(deg),MeasuredTCP[Xmm,Ymm,Zmm,Rx,Ry,Rz], for Pose Comparison",
    )
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(curr_dir)
    config_dir_path = os.path.join(project_root, "configs")
    c_path = os.path.join(config_dir_path, "calibration.conf")
    u_path = os.path.join(config_dir_path, "urcontrol.conf.UR5")

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
