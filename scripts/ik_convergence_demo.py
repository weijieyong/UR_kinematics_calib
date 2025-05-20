#!/usr/bin/env python3
"""
Demo: Improved Convergence Rate of Analytical IK Seed vs Random Seed

This script compares the convergence rate of the UR5 inverse kinematics solver when seeded with:
1. Analytical IK solution (closest branch)
2. Random joint values within joint limits

It generates a number of random target poses, attempts to solve IK for each using both seed strategies, and reports the convergence rates and statistics.

Usage:
    uv run  scripts/ik_convergence_demo.py [-n NUM_TESTS] [--joint-min-deg MIN] [--joint-max-deg MAX] [--seed SEED]

Requires: calibration.conf and urcontrol.conf.UR5 in the configs/ directory.
"""

import sys
import logging
import argparse
from pathlib import Path
import numpy as np

from ur_kinematics_calib.util import load_calibration, load_urcontrol_config
from ur_kinematics_calib.fk import fk_to_flange, tcp_transform
from ur_kinematics_calib.ik import ik_quik, analytic_ik_nominal, select_branch


def wrap_angles_rad(angles):
    angles = np.asarray(angles)
    return ((angles + np.pi) % (2 * np.pi)) - np.pi


def angular_diff(a, b):
    raw = b - a
    return (raw + np.pi) % (2 * np.pi) - np.pi


def calculate_pose_error(T1, T2):
    pos_error = T1[:3, 3] - T2[:3, 3]
    pos_error_norm = np.linalg.norm(pos_error) * 1000  # mm
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    if trace > 3 - 1e-10:
        rot_error_rad = 0.0
    else:
        rot_error_rad = np.arccos((trace - 1) / 2)
    rot_error_deg = np.rad2deg(rot_error_rad)
    return pos_error_norm, rot_error_deg


def main():
    parser = argparse.ArgumentParser(
        description="Demo: IK convergence with analytic vs random seed"
    )
    parser.add_argument(
        "-n",
        "--num-tests",
        type=int,
        default=100,
        help="Number of random poses to test",
    )
    parser.add_argument(
        "--joint-min-deg", type=float, default=-180.0, help="Min joint angle (deg)"
    )
    parser.add_argument(
        "--joint-max-deg", type=float, default=180.0, help="Max joint angle (deg)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    calib_path = config_dir / "calibration.conf"
    urcontrol_path = config_dir / "urcontrol.conf.UR5"
    if not (calib_path.exists() and urcontrol_path.exists()):
        print(f"Config files not found in {config_dir}")
        sys.exit(1)

    dt, da, dd, dalpha = load_calibration(calib_path)
    (a0, d0, alpha0, q_home0, j_dir), tcp_conf = load_urcontrol_config(urcontrol_path)
    eff_a = a0 + da
    eff_d = d0 + dd
    eff_alpha = alpha0 + dalpha
    T_fl_tcp = tcp_transform(tcp_conf)

    stats = {
        "analytic": {"success": 0, "iters": [], "fail": 0},
        "random": {"success": 0, "iters": [], "fail": 0},
    }
    pos_thresh_mm = 0.5
    rot_thresh_deg = 0.1

    for i in range(args.num_tests):
        q_gt = np.random.uniform(args.joint_min_deg, args.joint_max_deg, size=6)
        q_gt_rad = np.deg2rad(q_gt)
        dh_thetas = q_gt_rad + dt
        T_base_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas)
        T_target = T_base_fl @ T_fl_tcp

        # Analytic seed
        q_branches = analytic_ik_nominal(T_target @ np.linalg.inv(T_fl_tcp))
        if q_branches:
            q_branches_wrapped = [wrap_angles_rad(q) for q in q_branches]
            seed_ana = select_branch(q_branches_wrapped, q_gt_rad)
        else:
            seed_ana = q_gt_rad.copy()
        # Random seed
        seed_rand = np.random.uniform(
            np.deg2rad(args.joint_min_deg), np.deg2rad(args.joint_max_deg), size=6
        )

        for label, seed in [("analytic", seed_ana), ("random", seed_rand)]:
            q_sol, extra_data = ik_quik(
                eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, q_init=seed
            )
            e_sol, iterations, reason = extra_data
            q_sol = wrap_angles_rad(q_sol)
            sol_dh_thetas = q_sol + dt
            T_sol_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, sol_dh_thetas)
            T_sol = T_sol_fl @ T_fl_tcp
            pos_error_mm, rot_error_deg = calculate_pose_error(T_target, T_sol)
            converged = reason == "BREAKREASON_TOLERANCE"
            if (
                converged
                and pos_error_mm < pos_thresh_mm
                and rot_error_deg < rot_thresh_deg
            ):
                stats[label]["success"] += 1
                stats[label]["iters"].append(iterations)
            else:
                stats[label]["fail"] += 1
            if args.verbose:
                print(
                    f"Test {i + 1:03d} [{label}] - Converged: {converged}, Pos err: {pos_error_mm:.3f} mm, Rot err: {rot_error_deg:.3f} deg, Iters: {iterations}"
                )

    print("\n=== IK Convergence Rate Demo ===")
    for label in ["analytic", "random"]:
        total = stats[label]["success"] + stats[label]["fail"]
        rate = 100.0 * stats[label]["success"] / total if total > 0 else 0.0
        mean_iters = np.mean(stats[label]["iters"]) if stats[label]["iters"] else 0
        print(
            f"{label.title()} seed: Success {stats[label]['success']}/{total} ({rate:.2f}%), Mean iters: {mean_iters:.2f}"
        )
    print("Success = converged with pos error < 0.5mm and rot error < 0.1 deg.")


if __name__ == "__main__":
    main()
