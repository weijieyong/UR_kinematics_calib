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
from ur_kinematics_calib.ik import ik_quik

# Ensure project root is on path to import package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

RANDOM_OFFSET = 0.9


# Compute per-joint error in radians, wrapped to [–π, π]
def angular_diff(a, b):
    raw = b - a
    return (raw + np.pi) % (2 * np.pi) - np.pi


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
    # q_init = np.random.uniform(
    #     low=-np.pi, high=np.pi, size=6
    # )  # random initial guess, for testing
    # q_init = np.deg2rad(np.array([87.84, -110.6, 110.71, -72.87, -68.8, -100.12]))  # testing with some closer initial guess
    # q_init = q_home0  # use home joint angles as initial guess

    # Set q_init as a small random offset from parsed joint values to test robustness
    # Generate small random offsets for each joint independently
    random_offset = np.random.uniform(
        low=-RANDOM_OFFSET, high=RANDOM_OFFSET, size=6
    )  # Small random offsets in radians
    q_init = q_orig + random_offset
    logging.debug("Random joint offset (rad): %s", np.round(random_offset, 6).tolist())

    dh_thetas = q_orig + dt
    T_base_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas)
    T_target = T_base_fl @ T_fl_tcp

    # Solve IK starting from original joint angles
    q_sol, extra_data = ik_quik(
        eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, q_init=q_init
    )
    e_sol, iterations, reason, is_reachable = extra_data

    # Compare original vs solved joints (degrees)
    diff_rad = angular_diff(q_orig, q_sol)

    # Convert to degrees
    diff_deg = np.rad2deg(diff_rad)
    orig_deg = np.rad2deg(q_orig)
    sol_deg = np.rad2deg(q_sol)
    q_init_deg = np.rad2deg(q_init)

    logging.info("Initial guess (deg):        %s", np.round(q_init_deg, 6).tolist())
    logging.info("Original joints (deg):      %s", np.round(orig_deg, 6).tolist())
    logging.info("IK solution joints (deg):   %s", np.round(sol_deg, 6).tolist())
    logging.info("Joint errors (deg):         %s", np.round(diff_deg, 6).tolist())
    logging.info("Joint-space norm error (°): %.3e", np.linalg.norm(diff_deg))
    logging.info("Max joint error (°):        %.3e", np.max(np.abs(diff_deg)))
    logging.info("IK error:                   %.3e", np.linalg.norm(e_sol))
    logging.info("IK iterations:              %d", iterations)
    logging.info("IK status:                  %s", reason)
    return 0


if __name__ == "__main__":
    exit(main())
