#!/usr/bin/env python3
"""Forward Kinematics demo for UR5 using calibrated DH parameters."""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from ur_kinematics_calib.util import load_calibration, load_urcontrol_config
from ur_kinematics_calib.fk import fk_to_flange, tcp_transform
import logging

# Ensure project root is on path to import package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(description="UR5 FK Demo")

    joint_input_group = parser.add_mutually_exclusive_group()
    joint_input_group.add_argument(
        "-j",
        "--joints",
        type=str,
        help="Comma-separated 6 joint angles (deg) for FK. Defaults to home configuration.",
    )
    joint_input_group.add_argument(
        "-r",
        "--random-joints",
        action="store_true",
        help="Use random joint angles between -2*pi and 2*pi for FK.",
    )

    parser.add_argument(
        "--no-quik",
        action="store_true",
        help="Use Python FK implementation instead of QuIK",
    )
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level)",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # Load config files
    config_dir = os.path.join(project_root, "configs")
    calib_path = os.path.join(config_dir, "calibration.conf")
    urcontrol_path = os.path.join(config_dir, "urcontrol.conf.UR5")
    if not (os.path.exists(calib_path) and os.path.exists(urcontrol_path)):
        logging.error("Config files not found in configs directory under project root.")
        sys.exit(1)

    # Parse calibration & UR control
    dt, da, dd, dalpha = load_calibration(calib_path)
    (a0, d0, alpha0, q_home0, j_dir), tcp_conf = load_urcontrol_config(urcontrol_path)
    eff_a = a0 + da
    eff_d = d0 + dd
    eff_alpha = alpha0 + dalpha
    
    logging.debug(f"Effective DH a: {eff_a.tolist()}")
    logging.debug(f"Effective DH d: {eff_d.tolist()}")
    logging.debug(f"Effective DH alpha: {eff_alpha.tolist()}")
    logging.debug(f"Effective DH theta offsets: {dt.tolist()}")
    logging.debug(f"Joint directions: {j_dir.tolist()}")
    logging.debug(f"Home joint configuration (rad): {q_home0.tolist()}")
    logging.debug(f"TCP configuration: {tcp_conf.tolist()}")

    # Determine D-H joint thetas
    if args.joints:
        try:
            q_deg = np.array([float(x) for x in args.joints.split(",")])
            if q_deg.size != 6:
                raise ValueError
            logging.debug(f"Input joint angles (deg): {q_deg.tolist()}")
            dh_thetas = np.deg2rad(q_deg) + dt
        except Exception:
            logging.error("Invalid joint input: expected 6 comma-separated floats")
            sys.exit(1)
    elif args.random_joints:
        q_rad = np.random.uniform(-2 * np.pi, 2 * np.pi, 6)
        logging.info(f"Randomly generated joint angles (deg): {np.rad2deg(q_rad).round(4).tolist()}")
        dh_thetas = q_rad + dt
    else:
        logging.info("Using home joint configuration.")
        dh_thetas = q_home0 + dt
    
    logging.debug(f"Calculated DH thetas (rad): {dh_thetas.round(6).tolist()}")

    # Compute FK
    T_base_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas, use_quik=not args.no_quik)
    T_fl_tcp = tcp_transform(tcp_conf)
    T_base_tcp = T_base_fl @ T_fl_tcp

    # Extract pose
    pos_mm = T_base_tcp[0:3, 3] * 1000.0
    rotvec_rad = R_scipy.from_matrix(T_base_tcp[0:3, 0:3]).as_rotvec()

    # Display
    logging.info("Calculated TCP pose:")
    logging.info(f"  Position (mm): {pos_mm.round(4).tolist()}")
    logging.info(f"  Rotation vector (rad): {rotvec_rad.round(6).tolist()}")
    if not args.no_quik:
        logging.info("(Using QuIK FK solver)")
    else:
        logging.info("(Using Python FK solver)")


if __name__ == "__main__":
    main()
