#!/usr/bin/env python3
"""UR5 FK-IK Round Trip Test: 

This script performs the following steps:
1. Generate ground-truth poses from random joint configurations
2. Build seed set (two seeds per pose: continuous motion and analytic IK)
3. Run numerical IK for each seed
4. Evaluate success based on pose error thresholds
5. Track branch consistency
6. Collect statistics over multiple random poses
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ur_kinematics_calib.util import load_calibration, load_urcontrol_config
from ur_kinematics_calib.fk import fk_to_flange, tcp_transform
from ur_kinematics_calib.ik import ik_quik, analytic_ik_nominal, select_branch

# Constants for error thresholds
POSITION_ERROR_THRESHOLD_MM = 0.5
ROTATION_ERROR_THRESHOLD_DEG = 0.1
BRANCH_THRESHOLD_RAD = np.pi / 2
RANDOM_NUDGE_RANGE_RAD = 0.05


def wrap_angles_deg(angles: np.ndarray) -> np.ndarray:
    """Wrap a list or array of angles in degrees to [-180, 180)."""
    angles = np.asarray(angles)
    return ((angles + 180) % 360) - 180


def wrap_angles_rad(angles: np.ndarray) -> np.ndarray:
    """Wrap a list or array of angles in radians to [-π, π)."""
    angles = np.asarray(angles)
    return ((angles + np.pi) % (2 * np.pi)) - np.pi


def angular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-joint error in radians, wrapped to [–π, π]."""
    raw = b - a
    return (raw + np.pi) % (2 * np.pi) - np.pi


def calculate_pose_error(T1: np.ndarray, T2: np.ndarray) -> Tuple[float, float]:
    """Calculate position and orientation errors between two transformation matrices.
    Args:
        T1, T2: 4x4 homogeneous transformation matrices
    Returns:
        pos_error_norm: Euclidean distance error in position (mm)
        rot_error_deg: Rotation error in degrees
    """
    pos_error = T1[:3, 3] - T2[:3, 3]
    pos_error_norm = np.linalg.norm(pos_error) * 1000  # Convert to mm
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


def is_same_branch(
    q1: np.ndarray, q2: np.ndarray, threshold: float = BRANCH_THRESHOLD_RAD
) -> bool:
    """Check if two joint configurations are in the same branch."""
    diffs = np.abs(angular_diff(q1, q2))
    return np.all(diffs < threshold)


def generate_random_joints(
    num_tests: int, joint_min_deg: float, joint_max_deg: float
) -> List[np.ndarray]:
    """Generate random joint configurations within the specified range."""
    return [
        np.random.uniform(low=joint_min_deg, high=joint_max_deg, size=6)
        for _ in range(num_tests)
    ]


def process_joint_config(
    q_deg: np.ndarray,
    eff_a: np.ndarray,
    eff_alpha: np.ndarray,
    eff_d: np.ndarray,
    j_dir: np.ndarray,
    dt: np.ndarray,
    T_fl_tcp: np.ndarray,
    verbose_logging: bool,
) -> Dict[str, Dict[str, Any]]:
    """Processes a single joint configuration with two different seed strategies."""
    q_orig = np.deg2rad(q_deg)
    dh_thetas = q_orig + dt
    T_base_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas)
    T_target = T_base_fl @ T_fl_tcp
    random_nudge = np.random.uniform(
        -RANDOM_NUDGE_RANGE_RAD, RANDOM_NUDGE_RANGE_RAD, size=6
    )
    seed_prev = q_orig.copy() + random_nudge
    q_branches = analytic_ik_nominal(T_target @ np.linalg.inv(T_fl_tcp))
    if q_branches:
        q_branches_wrapped = [
            np.deg2rad(wrap_angles_deg(np.rad2deg(q))) for q in q_branches
        ]
        seed_ana = select_branch(q_branches_wrapped, q_orig)
    else:
        seed_ana = q_orig.copy()
    if verbose_logging:
        logging.debug(
            f"Original joints (deg): {np.round(np.rad2deg(q_orig), 3).tolist()}"
        )
        logging.debug(f"Seed_prev (deg): {np.round(np.rad2deg(seed_prev), 3).tolist()}")
        logging.debug(f"Seed_ana (deg): {np.round(np.rad2deg(seed_ana), 3).tolist()}")
    results = {}
    seeds = {"prev": seed_prev, "ana": seed_ana}
    for seed_name, seed_value in seeds.items():
        q_sol, extra_data = ik_quik(
            eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, q_init=seed_value
        )
        e_sol, iterations, reason = extra_data
        q_sol = wrap_angles_rad(q_sol)
        diff_rad = angular_diff(q_orig, q_sol)
        diff_deg = np.rad2deg(diff_rad)
        joint_error_norm_deg = np.linalg.norm(diff_deg)
        sol_dh_thetas = q_sol + dt
        T_sol_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, sol_dh_thetas)
        T_sol = T_sol_fl @ T_fl_tcp
        pos_error_mm, rot_error_deg = calculate_pose_error(T_target, T_sol)
        pos_error_ok = pos_error_mm < POSITION_ERROR_THRESHOLD_MM
        rot_error_ok = rot_error_deg < ROTATION_ERROR_THRESHOLD_DEG
        converged = reason == "BREAKREASON_TOLERANCE"
        is_success = pos_error_ok and rot_error_ok and converged
        same_branch = is_same_branch(q_orig, q_sol)
        if is_success:
            status_text_str = (
                f"OK: Pos_err={pos_error_mm:.3f}mm, Rot_err={rot_error_deg:.3f}°"
            )
            current_style = "green"
        else:
            fail_reasons = []
            if not converged:
                fail_reasons.append(f"IK: {reason}")
            if not pos_error_ok:
                fail_reasons.append(
                    f"Pos_err={pos_error_mm:.3f}mm > {POSITION_ERROR_THRESHOLD_MM}mm"
                )
            if not rot_error_ok:
                fail_reasons.append(
                    f"Rot_err={rot_error_deg:.3f}° > {ROTATION_ERROR_THRESHOLD_DEG}°"
                )
            status_text_str = " & ".join(fail_reasons)
            current_style = "red"
        results[seed_name] = {
            "q_orig_deg": np.rad2deg(q_orig),
            "q_seed_deg": np.rad2deg(seed_value),
            "q_sol_deg": np.rad2deg(q_sol),
            "diff_deg": diff_deg,
            "joint_error_norm_deg": joint_error_norm_deg,
            "pos_error_mm": pos_error_mm,
            "rot_error_deg": rot_error_deg,
            "iterations": iterations,
            "status_text_str": status_text_str,
            "current_style": current_style,
            "is_success": is_success,
            "same_branch": same_branch,
            "reason": reason,
            "q_branches": q_branches,
        }
    return results


def main() -> int:
    """Main entry point for UR5 IK/FK performance testing."""
    parser = argparse.ArgumentParser(
        description="UR5 IK/FK Performance Testing - Implements test plan from docs/test-plan.md"
    )
    parser.add_argument(
        "-n",
        "--num-tests",
        type=int,
        default=10,
        help="Number of random joint configurations to generate and test",
    )
    parser.add_argument(
        "--joint-min-deg",
        type=float,
        default=-180.0,
        help="Minimum joint angle in degrees (default: -180°)",
    )
    parser.add_argument(
        "--joint-max-deg",
        type=float,
        default=180.0,
        help="Maximum joint angle in degrees (default: 180°)",
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
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Print only success rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s: %(message)s",
        )

    config_dir = args.config_dir
    calib_path = config_dir / "calibration.conf"
    urcontrol_path = config_dir / "urcontrol.conf.UR5"
    if not (calib_path.exists() and urcontrol_path.exists()):
        logging.error(
            f"Config files not found in {config_dir}. Ensure calibration.conf and urcontrol.conf.UR5 exist."
        )
        return 1

    dt, da, dd, dalpha = load_calibration(calib_path)
    (a0, d0, alpha0, q_home0, j_dir), tcp_conf = load_urcontrol_config(urcontrol_path)
    eff_a = a0 + da
    eff_d = d0 + dd
    eff_alpha = alpha0 + dalpha
    T_fl_tcp = tcp_transform(tcp_conf)

    joint_configs = generate_random_joints(
        args.num_tests, args.joint_min_deg, args.joint_max_deg
    )

    if not joint_configs:
        logging.error("Failed to generate any joint configurations")
        return 1

    if not args.quiet:
        print(f"Generated {len(joint_configs)} random joint configurations for testing")

    stats = {
        "prev": {
            "processed": 0,
            "success": 0,
            "branch_flip": 0,
            "iterations": [],
            "pos_errors": [],
            "rot_errors": [],
        },
        "ana": {
            "processed": 0,
            "success": 0,
            "branch_flip": 0,
            "iterations": [],
            "pos_errors": [],
            "rot_errors": [],
        },
    }

    for i, q_deg in enumerate(joint_configs):
        results = process_joint_config(
            q_deg,
            eff_a,
            eff_alpha,
            eff_d,
            j_dir,
            dt,
            T_fl_tcp,
            args.verbose,
        )
        for seed_type, result in results.items():
            stats[seed_type]["processed"] += 1
            if result["is_success"]:
                stats[seed_type]["success"] += 1
            stats[seed_type]["iterations"].append(result["iterations"])
            stats[seed_type]["pos_errors"].append(result["pos_error_mm"])
            stats[seed_type]["rot_errors"].append(result["rot_error_deg"])
            if not result["same_branch"]:
                stats[seed_type]["branch_flip"] += 1
            if not args.quiet and not result["is_success"]:
                print(f"Test #{i + 1} - Seed type: {seed_type}")
                print(f"  q_orig (deg): {np.round(result['q_orig_deg'], 3).tolist()}")
                print(f"  q_seed (deg): {np.round(result['q_seed_deg'], 3).tolist()}")
                print(f"  q_sol  (deg): {np.round(result['q_sol_deg'], 3).tolist()}")
                print(
                    f"  Joint Error (deg): {np.round(result['diff_deg'], 3).tolist()}"
                )
                print(f"  Position Error: {result['pos_error_mm']:.3f} mm")
                print(f"  Rotation Error: {result['rot_error_deg']:.3f} °")
                print(f"  Iterations: {result['iterations']}")
                print(f"  Same Branch: {result['same_branch']}")
                print(f"  IK Status: {result['status_text_str']}")
                q_branches = result.get("q_branches", [])
                print("  All analytic IK branches (deg):")
                if q_branches:
                    q_branches_wrapped = [
                        np.deg2rad(wrap_angles_deg(np.rad2deg(q))) for q in q_branches
                    ]
                    for idx, q in enumerate(q_branches_wrapped):
                        print(
                            f"    Branch {idx + 1}: {np.round(np.rad2deg(q), 3).tolist()}"
                        )
                else:
                    print("    No branches found.")
                print("-" * 60)

    if stats["prev"]["processed"] > 0 and stats["ana"]["processed"] > 0:
        summary = {}
        for seed_type, data in stats.items():
            proc_count = data["processed"]
            if proc_count == 0:
                continue
            success_rate = (data["success"] / proc_count) * 100
            branch_flip_rate = (data["branch_flip"] / proc_count) * 100
            mean_iterations = np.mean(data["iterations"]) if data["iterations"] else 0
            mean_pos_error = np.mean(data["pos_errors"]) if data["pos_errors"] else 0
            mean_rot_error = np.mean(data["rot_errors"]) if data["rot_errors"] else 0
            summary[seed_type] = {
                "success_rate": success_rate,
                "branch_flip_rate": branch_flip_rate,
                "mean_iterations": mean_iterations,
                "mean_pos_error": mean_pos_error,
                "mean_rot_error": mean_rot_error,
            }
        if args.quiet:
            for seed_type, data in summary.items():
                seed_name = "Continuous" if seed_type == "prev" else "Analytic"
                print(f"{seed_name} Seed Success Rate: {data['success_rate']:.2f}%")
        else:
            print("\n" + "=" * 80)
            print(f"IK PERFORMANCE SUMMARY ({args.num_tests} random poses)")
            print("=" * 80)
            headers = [
                "Seed Type",
                "Success Rate",
                "Branch Flip Rate",
                "Mean Iters",
                "Mean Pos Err",
                "Mean Rot Err",
            ]
            print(
                f"{headers[0]:<12} {headers[1]:<15} {headers[2]:<18} {headers[3]:<12} {headers[4]:<15} {headers[5]:<12}"
            )
            print("-" * 80)
            seed_labels = {"prev": "Continuous", "ana": "Analytic"}
            for seed_type, data in summary.items():
                label = seed_labels[seed_type]
                success = f"{data['success_rate']:.2f}% ({stats[seed_type]['success']}/{stats[seed_type]['processed']})"
                branch = f"{data['branch_flip_rate']:.2f}% ({stats[seed_type]['branch_flip']}/{stats[seed_type]['processed']})"
                iters = f"{data['mean_iterations']:.2f}"
                pos_err = f"{data['mean_pos_error']:.4f} mm"
                rot_err = f"{data['mean_rot_error']:.4f}°"
                print(
                    f"{label:<12} {success:<15} {branch:<18} {iters:<12} {pos_err:<15} {rot_err:<12}"
                )
            print("=" * 80)
            print(
                f"Success criteria: Position error < {POSITION_ERROR_THRESHOLD_MM} mm, Orientation error < {ROTATION_ERROR_THRESHOLD_DEG}°"
            )
            print("Branch flip is not counted as failure, only tracked for analysis")
    else:
        if not args.quiet:
            print("\nNo IK attempts were processed to calculate statistics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
