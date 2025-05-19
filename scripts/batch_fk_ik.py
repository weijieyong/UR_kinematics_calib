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
import csv
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


def process_csv_row(
    row_str_list,
    eff_a,
    eff_alpha,
    eff_d,
    j_dir,
    dt,
    T_fl_tcp,
    verbose_logging,
    random_offset_magnitude,
):
    """Processes a single row of joint data from the CSV."""
    try:
        q_deg_list = [float(x.strip()) for x in row_str_list[:6]]
        q_orig = np.deg2rad(np.array(q_deg_list))
    except ValueError as e:
        return {
            "error": f"Error parsing joint angles: {row_str_list[:6]}. Details: {e}"
        }

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
        "error": None,
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
    parser = argparse.ArgumentParser(description="UR5 IK/FK Check from CSV")
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=Path("data/joint-eef-data.csv"),
        help="Path to CSV file containing joint angles in degrees (first 6 columns, header expected)",
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
        "--random-offset",
        type=float,
        default=0.8,
        help="Magnitude of the random offset applied to initial joint angles for IK (default: 0.8 radians)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    console = Console()

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

    if not args.csv_file.exists():
        logging.error(f"CSV file not found: {args.csv_file}")
        return 1

    table = Table(title="IK Batch Processing Results", show_lines=True)
    table.add_column("Set #", justify="right", style="cyan", no_wrap=True)
    table.add_column("q_orig (°)", style="yellow")
    table.add_column("q_init (°)", style="blue")
    table.add_column("q_sol (°)", style="green")
    table.add_column("Error (°)", style="red")
    # table.add_column("Max Err (°)", justify="right", style="red")
    # table.add_column("Cart. Err", justify="right", style="magenta")
    table.add_column("Iter.", justify="right", style="cyan")
    table.add_column("IK Status", style="bold")

    with open(args.csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)  # Skip header
            logging.debug(f"Skipped header: {header}")
            data_rows = list(reader)  # Read all data rows to get a count
            num_data_rows = len(data_rows)
            if num_data_rows == 0:
                logging.error(
                    f"CSV file {args.csv_file} has no data rows after the header."
                )
                console.print(
                    f"[bold red]Error: CSV file {args.csv_file} has no data rows after the header.[/bold red]"
                )
                return 1
        except StopIteration:
            logging.error(f"CSV file {args.csv_file} is empty or has no header.")
            console.print(
                f"[bold red]Error: CSV file {args.csv_file} is empty or has no header.[/bold red]"
            )
            return 1

        processed_count = 0
        success_count = 0  # Initialize counter for successful IK
        with Progress(console=console) as progress:
            task = progress.add_task(
                "[cyan]Processing CSV rows...", total=num_data_rows
            )

            for i, row in enumerate(data_rows):
                if not row or len(row) < 6:
                    logging.warning(
                        f"Skipping malformed or empty row {i + 2} (1-indexed data rows, including header): {row}"
                    )
                    progress.update(task, advance=1)
                    continue

                row_data_str = row[:6]
                result = process_csv_row(
                    row_data_str,
                    eff_a,
                    eff_alpha,
                    eff_d,
                    j_dir,
                    dt,
                    T_fl_tcp,
                    args.verbose,
                    args.random_offset,
                )

                if result["error"]:
                    logging.error(
                        f"Error processing CSV row {i + 2}: {result['error']}"
                    )
                    progress.update(task, advance=1)
                    continue

                processed_count += 1
                if result["is_success"]:
                    success_count += 1

                status_text = Text(
                    result["status_text_str"], style=result["current_style"]
                )

                # Add to table if failed, or if successful and verbose mode is on
                if not result["is_success"] or (result["is_success"] and args.verbose):
                    table.add_row(
                        str(processed_count),
                        str(np.round(result["q_orig_deg"], 3).tolist()),
                        str(np.round(result["q_init_deg"], 3).tolist()),
                        str(np.round(result["q_sol_deg"], 3).tolist()),
                        str(np.round(result["diff_deg"], 3).tolist()),
                        str(result["iterations"]),
                        status_text,
                    )

                progress.update(task, advance=1)

        if processed_count == 0:
            logging.warning(
                f"No valid joint data successfully processed from {args.csv_file} after the header."
            )
            # The previous console print for "No valid joint data found" might be redundant if num_data_rows was 0 earlier.
            # This condition now means no rows were *successfully* processed, even if there were rows.
            if (
                num_data_rows > 0
            ):  # Only print if there were rows but none were valid for processing
                console.print(
                    f"[bold orange_red1]Warning: No valid joint data successfully processed from {args.csv_file} after the header.[/bold orange_red1]"
                )
            return 1

    console.print(table)

    if processed_count > 0:
        success_percentage = (success_count / processed_count) * 100
        console.print(
            f"\n[bold]IK Success Rate ({success_count}/{processed_count}):[/bold] [green]{success_percentage:.2f}%[/green] (met BREAKREASON_TOLERANCE AND Joint Error Norm < 1e-3 °)."
        )
    else:
        console.print(
            "\n[bold yellow]No IK attempts were processed to calculate a success rate.[/bold yellow]"
        )

    return 0


if __name__ == "__main__":
    exit(main())
