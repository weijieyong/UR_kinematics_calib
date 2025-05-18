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
    import quik_bind as quikpy
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure all required modules are installed and in the Python path.")
    print(f"Project root added to sys.path: {project_root}")
    sys.exit(1)


def init_quik_robot(eff_a, eff_alpha, eff_d):
    # Create DH parameter matrix for QuIK (a, alpha, d, theta_offset)
    dh = np.zeros((6, 4))
    dh[:, 0] = eff_a
    dh[:, 1] = eff_alpha
    dh[:, 2] = eff_d
    # All joints are revolute
    link_types = np.zeros(6, dtype=bool)  # False = revolute, True = prismatic
    quikpy.init_robot(dh, link_types)


def benchmark_python_fk(eff_a, eff_alpha, eff_d, j_dir, dt, num_iterations):
    fk_times = []
    random_joints = np.random.uniform(-np.pi, np.pi, (num_iterations, 6))
    
    for i in tqdm(range(num_iterations), desc="Python FK"):
        dh_thetas_fk = random_joints[i] + dt
        start_time = time.perf_counter()
        _ = fk_to_flange(eff_a, eff_alpha, eff_d, j_dir, dh_thetas_fk)
        end_time = time.perf_counter()
        fk_times.append(end_time - start_time)
    
    return fk_times


def benchmark_quik_fk(dt, num_iterations):
    fk_times = []
    random_joints = np.random.uniform(-np.pi, np.pi, (num_iterations, 6))
    
    for i in tqdm(range(num_iterations), desc="QuIK FK"):
        dh_thetas_fk = random_joints[i] + dt
        start_time = time.perf_counter()
        _ = quikpy.fkn(dh_thetas_fk)
        end_time = time.perf_counter()
        fk_times.append(end_time - start_time)
    
    return fk_times


def print_stats(times, method):
    times_ms = np.array(times) * 1000
    avg_time = np.mean(times_ms)
    min_time = np.min(times_ms)
    max_time = np.max(times_ms)
    std_time = np.std(times_ms)
    
    print(f"\n--- {method} Benchmark Results ---")
    print(f"Average calculation time: {avg_time:.4f} ms")
    print(f"Min calculation time: {min_time:.4f} ms")
    print(f"Max calculation time: {max_time:.4f} ms")
    print(f"Std Dev calculation time: {std_time:.4f} ms")


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
    (a0, d0, alpha0, q_home0, j_dir), _ = load_urcontrol_config(u_path)

    eff_a, eff_d, eff_alpha = a0 + da, d0 + dd, alpha0 + dalpha

    print(f"Starting FK benchmark comparison with {num_iterations} iterations...\n")
    print("Effective D-H (a, d in m; alpha in rad):")
    print(f"  a_eff:     {eff_a.round(6).tolist()}")
    print(f"  d_eff:     {eff_d.round(6).tolist()}")
    print(f"  alpha_eff: {eff_alpha.round(6).tolist()}\n")

    # Initialize QuIK robot
    init_quik_robot(eff_a, eff_alpha, eff_d)

    # Run benchmarks
    python_times = benchmark_python_fk(eff_a, eff_alpha, eff_d, j_dir, dt, num_iterations)
    quik_times = benchmark_quik_fk(dt, num_iterations)

    # Print results
    print_stats(python_times, "Python FK")
    print_stats(quik_times, "QuIK FK")

    # Calculate speedup
    avg_python = np.mean(python_times)
    avg_quik = np.mean(quik_times)
    speedup = avg_python / avg_quik
    print(f"\nQuIK speedup over Python: {speedup:.2f}x")


if __name__ == "__main__":
    main()
