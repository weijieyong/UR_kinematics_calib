#!/usr/bin/env python3
"""UR5 IK Demo: Solve inverse kinematics for UR5 using calibrated DH parameters."""

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
from ur_kinematics_calib.util import load_calibration, load_urcontrol_config
from ur_kinematics_calib.fk import fk_to_flange, tcp_transform
from ur_kinematics_calib.ik import ik_numerical

# Ensure project root is on path to import package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def parse_joints(joint_str: str) -> np.ndarray:
    parts = joint_str.split(",")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "Expected 6 comma-separated joint angles in degrees (q1,..,q6)"
        )
    values = [float(x) for x in parts]
    q_deg = np.array(values)
    return np.deg2rad(q_deg)


def main():
    parser = argparse.ArgumentParser(description="UR5 IK Demo")
    parser.add_argument(
        "-j",
        "--joints",
        type=parse_joints,
        required=True,
        help="Comma-separated 6-tuple joint angles in degrees",
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

    # Given original joint angles, compute FK to TCP pose
    q_orig = args.joints
    q_init_random = np.random.uniform(
        low=-np.pi, high=np.pi, size=6
    )  # random initial guess, for testing
    dh_thetas = q_orig + dt
    T_base_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas)
    T_target = T_base_fl @ T_fl_tcp

    # Solve IK starting from original joint angles
    q_sol, _ = ik_numerical(
        eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, q_init=q_home0
    )

    # Compare original vs solved joints
    diff = q_sol - q_orig
    logging.info("Original Joints (deg): %s", np.rad2deg(q_orig).round(6).tolist())
    logging.info("IK Solution Joints (deg): %s", np.rad2deg(q_sol).round(6).tolist())
    logging.info(
        "Joint Error (deg): %s, Norm: %.6f",
        np.rad2deg(diff).round(6).tolist(),
        np.linalg.norm(np.rad2deg(diff)),
    )
    return 0


if __name__ == "__main__":
    exit(main())
