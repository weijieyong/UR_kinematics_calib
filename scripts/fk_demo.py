#!/usr/bin/env python3
"""Forward Kinematics demo for UR5 using calibrated DH parameters."""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from ur_kinematics_calib.util import load_calibration, load_urcontrol_config
from ur_kinematics_calib.fk import fk_to_flange, tcp_transform

# Ensure project root is on path to import package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(description="UR5 FK Demo")
    parser.add_argument(
        "-j",
        "--joints",
        type=str,
        help="Comma-separated 6 joint angles (deg) for FK. Defaults to home configuration.",
    )
    parser.add_argument(
        "--no-quik",
        action="store_true",
        help="Use Python FK implementation instead of QuIK",
    )
    args = parser.parse_args()

    # Load config files
    config_dir = os.path.join(project_root, "configs")
    calib_path = os.path.join(config_dir, "calibration.conf")
    urcontrol_path = os.path.join(config_dir, "urcontrol.conf.UR5")
    if not (os.path.exists(calib_path) and os.path.exists(urcontrol_path)):
        print("Config files not found in configs directory under project root.")
        sys.exit(1)

    # Parse calibration & UR control
    dt, da, dd, dalpha = load_calibration(calib_path)
    (a0, d0, alpha0, q_home0, j_dir), tcp_conf = load_urcontrol_config(urcontrol_path)
    eff_a = a0 + da
    eff_d = d0 + dd
    eff_alpha = alpha0 + dalpha

    # Determine D-H joint thetas
    if args.joints:
        try:
            q_deg = np.array([float(x) for x in args.joints.split(",")])
            if q_deg.size != 6:
                raise ValueError
            dh_thetas = np.deg2rad(q_deg) + dt
        except Exception:
            print("Invalid joint input: expected 6 comma-separated floats")
            sys.exit(1)
    else:
        dh_thetas = q_home0 + dt

    # Compute FK
    T_base_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas, use_quik=not args.no_quik)
    T_fl_tcp = tcp_transform(tcp_conf)
    T_base_tcp = T_base_fl @ T_fl_tcp

    # Extract pose
    pos_mm = T_base_tcp[0:3, 3] * 1000.0
    rotvec_rad = R_scipy.from_matrix(T_base_tcp[0:3, 0:3]).as_rotvec()

    # Display
    print("Calculated TCP pose:")
    print(f"  Position (mm): {pos_mm.round(4).tolist()}")
    print(f"  Rotation vector (rad): {rotvec_rad.round(6).tolist()}")
    if not args.no_quik:
        print("(Using QuIK FK solver)")
    else:
        print("(Using Python FK solver)")


if __name__ == "__main__":
    main()
