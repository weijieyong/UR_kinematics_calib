#!/usr/bin/env python3
"""UR5 IK Demo: Solve inverse kinematics for UR5 using calibrated DH parameters."""

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from ur_kinematics_calib.util import load_calibration, load_urcontrol_config
from ur_kinematics_calib.fk import tcp_transform
from ur_kinematics_calib.ik import ik_numerical

# Ensure project root is on path to import package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def parse_pose(pose_str: str) -> np.ndarray:
    parts = pose_str.split(",")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "Expected 6 comma-separated values: Xmm,Ymm,Zmm,Rx,Ry,Rz"
        )
    values = [float(x) for x in parts]
    pos = np.array(values[:3]) / 1000.0
    rotvec = np.array(values[3:])
    T = np.eye(4)
    T[:3, 3] = pos
    if np.linalg.norm(rotvec) > 1e-9:
        T[:3, :3] = R_scipy.from_rotvec(rotvec).as_matrix()
    return T


def main():
    parser = argparse.ArgumentParser(description="UR5 IK Demo")
    parser.add_argument(
        "-p",
        "--pose",
        type=parse_pose,
        required=True,
        help="Comma-separated 6-tuple: Xmm,Ymm,Zmm,Rx,Ry,Rz",
    )
    parser.add_argument(
        "-c",
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs",
        help="Directory containing calibration.conf and urcontrol.conf.UR5",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable detailed logging"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load config files
    config_dir = args.config_dir
    calib_path = config_dir / "calibration.conf"
    urcontrol_path = config_dir / "urcontrol.conf.UR5"
    if not (calib_path.exists() and urcontrol_path.exists()):
        logging.error("Config files not found in configs directory under project root.")
        return 1

    # Parse calibration & UR control
    dt, da, dd, dalpha = load_calibration(calib_path)
    (a0, d0, alpha0, q_home0, j_dir), tcp_conf = load_urcontrol_config(urcontrol_path)
    eff_a = a0 + da
    eff_d = d0 + dd
    eff_alpha = alpha0 + dalpha

    T_fl_tcp = tcp_transform(tcp_conf)

    # Determine D-H joint thetas
    T_target = args.pose
    q_init = q_home0

    q_sol, _ = ik_numerical(
        eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, q_init
    )
    logging.info("IK Solution:")
    logging.info(f"  Joints (deg): {np.rad2deg(q_sol).round(6).tolist()}")
    return 0


if __name__ == "__main__":
    exit(main())
