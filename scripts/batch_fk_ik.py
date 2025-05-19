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
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.progress import Progress

# Ensure project root is on path to import package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)


# Compute per-joint error in radians, wrapped to [–π, π]
def angular_diff(a, b):
    raw = b - a
    return (raw + np.pi) % (2 * np.pi) - np.pi


def generate_random_joints(num_tests, joint_min_deg, joint_max_deg):
    """Generate random joint configurations within the specified range."""
    joint_configs = []

    for _ in range(num_tests):
        # Generate 6 random joint values within the specified range
        q_deg = np.random.uniform(low=joint_min_deg, high=joint_max_deg, size=6)
        joint_configs.append(q_deg)

    return joint_configs


def process_joint_config(
    q_deg,
    eff_a,
    eff_alpha,
    eff_d,
    j_dir,
    dt,
    T_fl_tcp,
    verbose_logging,
    random_offset_magnitude,
):
    """Processes a single joint configuration."""
    q_orig = np.deg2rad(q_deg)

    random_offset_val = np.random.uniform(
        low=-random_offset_magnitude, high=random_offset_magnitude, size=6
    )
    q_init = q_orig + random_offset_val
    if verbose_logging:
        logging.debug(
            "Random joint offset (rad): %s", np.round(random_offset_val, 6).tolist()
        )

    dh_thetas = q_orig + dt
    T_base_fl = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas)
    T_target = T_base_fl @ T_fl_tcp

    q_sol, extra_data = ik_quik(
        eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, q_init=q_init
    )
    e_sol, iterations, reason = extra_data

    diff_rad = angular_diff(q_orig, q_sol)
    diff_deg = np.rad2deg(diff_rad)
    orig_deg_display = np.rad2deg(q_orig)
    sol_deg_display = np.rad2deg(q_sol)
    q_init_deg_display = np.rad2deg(q_init)

    joint_error_norm_deg = np.linalg.norm(diff_deg)

    condA_met = reason == "BREAKREASON_TOLERANCE"
    condB_met = joint_error_norm_deg < 1e-3
    is_success = condA_met and condB_met

    status_text_str = ""
    current_style = ""

    if is_success:
        status_text_str = f"OK: {reason} & JntErr < 1e-3 ({joint_error_norm_deg:.1e}°)"
        current_style = "green"
    else:
        fail_reasons_list = []
        if not condA_met:
            fail_reasons_list.append(f"IK: {reason}")
        if not condB_met:
            fail_reasons_list.append(f"JntErr >= 1e-3 ({joint_error_norm_deg:.1e}°)")

        if condA_met and not condB_met:  # IK reason was fine, but joint error was high
            status_text_str = (
                f"IK: {reason} (but JntErr >= 1e-3 ({joint_error_norm_deg:.1e}°))"
            )
        else:
            status_text_str = " & ".join(fail_reasons_list)
        current_style = "red"

    return {
        "q_orig_deg": orig_deg_display,
        "q_init_deg": q_init_deg_display,
        "q_sol_deg": sol_deg_display,
        "diff_deg": diff_deg,
        "iterations": iterations,
        "status_text_str": status_text_str,
        "current_style": current_style,
        "is_success": is_success,
    }


def main():
    parser = argparse.ArgumentParser(
        description="UR5 IK/FK Check with Random Joint Values"
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
        "-ro",
        "--random-offset",
        type=float,
        default=0.8,
        help="Magnitude of the random offset applied to initial joint angles for IK (default: 0.8 radians)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results",
    )
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Configure logging - quiet mode takes precedence over verbose
    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s: %(message)s",
        )

    console = Console(quiet=args.quiet)

    # Load config files
    config_dir = args.config_dir
    calib_path = config_dir / "calibration.conf"
    urcontrol_path = config_dir / "urcontrol.conf.UR5"
    if not (calib_path.exists() and urcontrol_path.exists()):
        logging.error(
            f"Config files not found in {config_dir}. Ensure calibration.conf and urcontrol.conf.UR5 exist."
        )
        return 1

    # Parse calibration & UR control
    dt, da, dd, dalpha = load_calibration(calib_path)
    (a0, d0, alpha0, q_home0, j_dir), tcp_conf = load_urcontrol_config(urcontrol_path)
    eff_a = a0 + da
    eff_d = d0 + dd
    eff_alpha = alpha0 + dalpha
    T_fl_tcp = tcp_transform(tcp_conf)

    # Generate random joint configurations
    joint_configs = generate_random_joints(
        args.num_tests, args.joint_min_deg, args.joint_max_deg
    )

    if not joint_configs:
        logging.error("Failed to generate any joint configurations")
        return 1

    if not args.quiet:
        console.print(
            f"[bold cyan]Generated {len(joint_configs)} random joint configurations for testing[/bold cyan]"
        )

    table = Table(title="IK Batch Processing Results", show_lines=True)
    table.add_column("Test #", justify="right", style="cyan", no_wrap=True)
    table.add_column("q_orig (°)", style="yellow")
    table.add_column("q_init (°)", style="blue")
    table.add_column("q_sol (°)", style="green")
    table.add_column("Error (°)", style="red")
    table.add_column("Iter.", justify="right", style="cyan")
    table.add_column("IK Status", style="bold")

    processed_count = 0
    success_count = 0  # Initialize counter for successful IK

    # Use progress bar only in non-quiet mode
    if args.quiet:
        # Process without progress bar in quiet mode
        for i, q_deg in enumerate(joint_configs):
            result = process_joint_config(
                q_deg,
                eff_a,
                eff_alpha,
                eff_d,
                j_dir,
                dt,
                T_fl_tcp,
                args.verbose,
                args.random_offset,
            )

            processed_count += 1
            if result["is_success"]:
                success_count += 1
    else:
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[cyan]Processing joint configurations...", total=len(joint_configs)
            )

            for i, q_deg in enumerate(joint_configs):
                result = process_joint_config(
                    q_deg,
                    eff_a,
                    eff_alpha,
                    eff_d,
                    j_dir,
                    dt,
                    T_fl_tcp,
                    args.verbose,
                    args.random_offset,
                )

                processed_count += 1
                if result["is_success"]:
                    success_count += 1

                status_text = Text(
                    result["status_text_str"], style=result["current_style"]
                )

                # Add to table if failed, or if successful and verbose mode is on
                if not result["is_success"] or (result["is_success"] and args.verbose):
                    table.add_row(
                        str(i + 1),
                        str(np.round(result["q_orig_deg"], 3).tolist()),
                        str(np.round(result["q_init_deg"], 3).tolist()),
                        str(np.round(result["q_sol_deg"], 3).tolist()),
                        str(np.round(result["diff_deg"], 3).tolist()),
                        str(result["iterations"]),
                        status_text,
                    )

                progress.update(task, advance=1)

    if not args.quiet:
        console.print(table)

    if processed_count > 0:
        success_percentage = (success_count / processed_count) * 100
        if args.quiet:
            # In quiet mode, print in the format: IK Success Rate (X/Y): Z.ZZ%
            print(
                f"IK Success Rate ({success_count}/{processed_count}): {success_percentage:.2f}%"
            )
        else:
            console.print(
                f"\n[bold]IK Success Rate ({success_count}/{processed_count}):[/bold] [green]{success_percentage:.2f}%[/green] (met BREAKREASON_TOLERANCE AND Joint Error Norm < 1e-3 °)."
            )
    else:
        if not args.quiet:
            console.print(
                "\n[bold yellow]No IK attempts were processed to calculate a success rate.[/bold yellow]"
            )

    return 0


if __name__ == "__main__":
    exit(main())
