import time
import numpy as np
import os
import sys
from tqdm import tqdm

# Determine the project root directory and add it to sys.path
# Current script path: project_root/scripts/benchmark_fk.py
# Project root is two levels up from the script's directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


try:
    from ur_kinematics_calib.util import (
        load_calibration,
        load_urcontrol_config,
    )
    from ur_kinematics_calib.fk import fk_to_flange
except ImportError:
    print("Error: Could not import from ur_kinematics_calib.fk.py.")
    print(
        "Make sure 'ur_kinematics_calib' is in the project root and contains 'fk.py' and '__init__.py'."
    )
    print(f"Project root added to sys.path: {project_root}")
    sys.exit(1)


def main():
    num_iterations = 10000  # Number of FK calculations to average

    # Config files are in the 'configs' directory at the project root
    config_dir_path = os.path.join(project_root, "configs")
    c_path = os.path.join(config_dir_path, "calibration.conf")
    u_path = os.path.join(config_dir_path, "urcontrol.conf.UR5")

    if not (os.path.exists(c_path) and os.path.exists(u_path)):
        print("Error: Config file(s) not found in ./configs/")
        print(f"Expected calibration.conf at: {c_path}")
        print(f"Expected urcontrol.conf.UR5 at: {u_path}")
        sys.exit(1)

    dt, da, dd, dalpha = load_calibration(c_path)
    (a0, d0, alpha0, q_home0, j_dir), _ = load_urcontrol_config(
        u_path
    )  # tcp_conf not needed for fk_to_flange

    eff_a, eff_d, eff_alpha = a0 + da, d0 + dd, alpha0 + dalpha

    print(f"Starting FK benchmark with {num_iterations} iterations...\n")
    print("Effective D-H (a, d in m; alpha in rad):")
    print(f"  a_eff:     {eff_a.round(6).tolist()}")
    print(f"  d_eff:     {eff_d.round(6).tolist()}")
    print(f"  alpha_eff: {eff_alpha.round(6).tolist()}\n")

    fk_times = []

    for i in tqdm(range(num_iterations), desc="FK computations"):
        # Generate random joint angles (radians) between -pi and pi for 6 joints
        random_joints_rad = np.random.uniform(-np.pi, np.pi, 6)
        dh_thetas_fk = random_joints_rad + dt  # Apply calibration offsets

        start_time = time.perf_counter()
        _ = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas_fk)
        end_time = time.perf_counter()

        fk_times.append(end_time - start_time)

    average_time_ms = (sum(fk_times) / num_iterations) * 1000
    min_time_ms = min(fk_times) * 1000
    max_time_ms = max(fk_times) * 1000

    print("\n--- FK Benchmark Results ---")
    print(f"Number of iterations: {num_iterations}")
    print(f"Average FK calculation time: {average_time_ms:.4f} ms")
    print(f"Min FK calculation time: {min_time_ms:.4f} ms")
    print(f"Max FK calculation time: {max_time_ms:.4f} ms")


if __name__ == "__main__":
    main()
