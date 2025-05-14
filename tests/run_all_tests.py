#!/usr/bin/env python3
import csv
import subprocess
import sys
import os
import re

from rich.console import Console
from rich.table import Table

# Thresholds
THRESH_POS = 0.08  # mm
THRESH_ROT = 0.03  # deg


def main():
    # Locate data and script paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "joint-eef-data.csv")
    script_path = os.path.join(script_dir, "dry_parse.py")

    console = Console()
    table = Table(title="Calibration Test Results")  # build table object[2]
    table.add_column("Test", justify="left", style="cyan")
    table.add_column("Pos (mm)", justify="right")
    table.add_column("Pos OK", justify="center")
    table.add_column("Rot (°)", justify="right")
    table.add_column("Rot OK", justify="center")

    # Read CSV and gather results
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            test_name = ",".join(row)
            # Run the comparison script
            proc = subprocess.run(
                [sys.executable, script_path, "--compare", test_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Extract numeric errors
            pos_match = re.search(r"Norm \(mm\):\s*([0-9.\-]+)", proc.stdout)
            rot_match = re.search(r"Rot Error \(deg\):\s*([0-9.\-]+)", proc.stdout)
            if not pos_match or not rot_match:
                continue

            pos = float(pos_match.group(1))
            rot = float(rot_match.group(1))
            ok_pos = pos <= THRESH_POS
            ok_rot = rot <= THRESH_ROT

            # Color‐coded status using Rich markup[4]
            pos_flag = "[green]PASS[/]" if ok_pos else "[red]FAIL[/]"
            rot_flag = "[green]PASS[/]" if ok_rot else "[red]FAIL[/]"

            table.add_row(
                test_name,
                f"{pos:.4f}",
                pos_flag,
                f"{rot:.4f}",
                rot_flag,
            )

            # Print any stderr messages immediately
            if proc.stderr:
                console.print(f"[bold red]{proc.stderr.strip()}[/]")

    # Render the final table
    console.print(table)


if __name__ == "__main__":
    main()
