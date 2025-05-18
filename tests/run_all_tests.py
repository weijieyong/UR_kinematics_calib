#!/usr/bin/env python3
import csv
import subprocess
import sys
import os
import re

from rich.console import Console, Group
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.style import Style
from typing import Tuple

# Thresholds (set these depending on your application)
THRESH_POS = 0.08  # mm
THRESH_ROT = 0.05  # deg


def main():
    # Locate data and script paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "joint-eef-data.csv")
    script_path = os.path.join(script_dir, "dry_parse.py")

    console = Console()

    # Create a summary panel
    summary = {"total": 0, "passed": 0, "pos_errors": [], "rot_errors": []}

    # Create main results table
    table = Table(
        title="[bold]Calibration Test Results[/]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
        expand=True,
    )

    # Add columns with better formatting
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Joints (abbr.)", style="cyan")
    table.add_column("Pos (mm)", justify="right", style="bright_white")
    table.add_column("Pos Status", justify="center")
    table.add_column("Rot (°)", justify="right", style="bright_white")
    table.add_column("Rot Status", justify="center")

    def get_error_color(value: float, threshold: float) -> Tuple[str, str]:
        """Get color based on error value and threshold."""
        ratio = min(value / (threshold * 2), 1.0)  # Cap at 1.0 for very large errors
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        return f"#{r:02x}{g:02x}00"

    def format_error(value: float, threshold: float) -> Text:
        """Format error value with color based on threshold."""
        color = get_error_color(value, threshold)
        style = Style(color=color, bold=value > threshold)
        return Text(f"{value:.4f}", style=style)

    # Read CSV and gather results
    with open(csv_path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        # Skip header row
        next(reader, None)

        # Get total number of tests for progress tracking
        rows = list(reader)
        total_tests = len(rows)

        console.print(f"[bold]Running {total_tests} tests...[/]")

        for test_num, row in track(
            enumerate(rows, 1), total=total_tests, description="Testing..."
        ):
            if not row or len(row) < 6:  # Skip empty or invalid rows
                continue

            test_name = ",".join(row)
            try:
                # Convert first 6 values to floats and format for display
                joint_values = [f"{float(val):.2f}" for val in row[:6]]
                test_display = " ".join(joint_values)
            except (ValueError, IndexError):
                test_display = "Invalid data"

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

            # Update summary
            summary["total"] += 1
            if ok_pos and ok_rot:
                summary["passed"] += 1
            summary["pos_errors"].append(pos)
            summary["rot_errors"].append(rot)

            # Format status with icons and colors
            pos_status = Text(
                "✓" if ok_pos else "✗", style="green" if ok_pos else "red"
            )
            rot_status = Text(
                "✓" if ok_rot else "✗", style="green" if ok_rot else "red"
            )

            # Add row with color-coded errors
            table.add_row(
                str(test_num),
                test_display,
                format_error(pos, THRESH_POS),
                pos_status,
                format_error(rot, THRESH_ROT),
                rot_status,
            )

            # Print any stderr messages immediately
            if proc.stderr:
                console.print(f"[bold red]{proc.stderr.strip()}[/]")

    # Calculate statistics
    avg_pos_error = (
        sum(summary["pos_errors"]) / len(summary["pos_errors"])
        if summary["pos_errors"]
        else 0
    )
    max_pos_error = max(summary["pos_errors"]) if summary["pos_errors"] else 0
    avg_rot_error = (
        sum(summary["rot_errors"]) / len(summary["rot_errors"])
        if summary["rot_errors"]
        else 0
    )
    max_rot_error = max(summary["rot_errors"]) if summary["rot_errors"] else 0
    pass_rate = (
        (summary["passed"] / summary["total"]) * 100 if summary["total"] > 0 else 0
    )

    # Create summary panel
    summary_panel = Panel(
        Group(
            f"[bold]Tests Run:[/] {summary['total']}",
            f"[bold green]Passed:[/] {summary['passed']} ({pass_rate:.1f}%)",
            f"[bold]Position Errors (mm):[/] Avg: {avg_pos_error:.4f}, Max: {max_pos_error:.4f}",
            f"[bold]Rotation Errors (°):[/]  Avg: {avg_rot_error:.4f}, Max: {max_rot_error:.4f}",
        ),
        title="[bold]Test Summary[/]",
        border_style="green"
        if pass_rate == 100
        else "yellow"
        if pass_rate >= 80
        else "red",
        padding=(1, 2),
    )

    # Display results
    console.print()
    console.print(summary_panel)
    console.print()
    console.print(table)

    # Add final status message
    if pass_rate == 100:
        console.print("\n[bold green]✓ All tests passed![/]")
    else:
        console.print(
            f"\n[bold {'yellow' if pass_rate >= 80 else 'red'}]⚠ {summary['total'] - summary['passed']} test(s) failed[/]"
        )


if __name__ == "__main__":
    main()
