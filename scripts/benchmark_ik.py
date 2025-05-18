import time
import numpy as np
import os
import sys
import logging
import argparse
from tqdm import tqdm

# Determine the project root directory and add it to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Benchmark IK performance")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose logging"
)
parser.add_argument(
    "-n", "--iterations", type=int, default=500, help="Number of iterations for benchmark"
)
args = parser.parse_args()
logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(levelname)s: %(message)s",
)

try:
    from ur_kinematics_calib.util import (
        load_calibration,
        load_urcontrol_config,
    )
    from ur_kinematics_calib.fk import fk_to_flange, tcp_transform
    from ur_kinematics_calib.ik import ik_numerical, ik_quik

    logging.info("Successfully imported UR kinematics modules")
except ImportError:
    logging.error("Error: Could not import from ur_kinematics_calib.")
    logging.error(
        "Make sure 'ur_kinematics_calib' is in the project root and contains '__init__.py'."
    )
    logging.error(f"Project root added to sys.path: {project_root}")
    sys.exit(1)


def run_ik_benchmark(solver_name, solver_func, num_iterations, eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, q_home0, max_ik_time_seconds=1.0):
    """Run benchmark for a specific IK solver and return metrics."""
    logging.info(f"Starting benchmark for {solver_name} with {num_iterations} iterations...")
    
    ik_times = []
    successful_iks = 0
    failed_iks = 0
    joint_errors = []
    iterations_count = []
    
    for i in tqdm(range(num_iterations), desc=f"{solver_name} iterations"):
        # 1. Generate random joint angles to create a reachable target pose
        random_joints_rad_for_target = np.random.uniform(-np.pi, np.pi, 6)
        dh_thetas_for_target = random_joints_rad_for_target + dt

        # 2. Calculate FK to get the target TCP pose
        T_base_fl_target = fk_to_flange(
            eff_a, eff_alpha, eff_d, j_dir, dh_thetas_for_target
        )
        T_target_tcp = T_base_fl_target @ T_fl_tcp

        # 3. Use q_home0 as the initial guess for IK
        q_init_ik = q_home0.copy()

        start_time = time.perf_counter()
        try:
            # Run the solver
            q_sol, result = solver_func(
                eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target_tcp, q_init_ik
            )
            end_time = time.perf_counter()
            duration = end_time - start_time
            ik_times.append(duration)
            
            # Check if solver was successful based on solver type
            if solver_name == "IK Numerical":
                success = result.success and duration < max_ik_time_seconds
                if success:
                    successful_iks += 1
                    iterations_count.append(result.nfev)  # number of function evaluations
                else:
                    failed_iks += 1
                    if not result.success:
                        logging.debug(f"Iteration {i + 1}: IK failed to converge. Cost: {result.cost:.2e}")
                    if duration >= max_ik_time_seconds:
                        logging.debug(f"Iteration {i + 1}: IK timed out ({duration:.2f}s).")
            elif solver_name == "IK QuIK":
                e_sol, iters, reason = result
                error_norm = np.linalg.norm(e_sol)
                success = error_norm < 1e-3 and duration < max_ik_time_seconds
                iterations_count.append(iters)
                if success:
                    successful_iks += 1
                else:
                    failed_iks += 1
                    if error_norm >= 1e-3:
                        logging.debug(f"Iteration {i + 1}: QuIK error too large: {error_norm:.2e}")
                    if duration >= max_ik_time_seconds:
                        logging.debug(f"Iteration {i + 1}: QuIK timed out ({duration:.2f}s).")
                    
        except Exception as e:
            end_time = time.perf_counter()  # still record time up to failure
            ik_times.append(end_time - start_time)
            failed_iks += 1
            logging.debug(f"Iteration {i + 1}: {solver_name} raised an exception: {e}")

        # Compare IK solution to original random joints
        error = q_sol - random_joints_rad_for_target
        joint_errors.append(error)

    # Calculate metrics
    if ik_times:  # Ensure ik_times is not empty
        average_time_ms = (sum(ik_times) / len(ik_times)) * 1000
        min_time_ms = min(ik_times) * 1000
        max_time_ms = max(ik_times) * 1000
        success_rate = (successful_iks / num_iterations) * 100
    else:
        average_time_ms = min_time_ms = max_time_ms = float("nan")
        success_rate = 0.0
        
    avg_iterations = np.mean(iterations_count) if iterations_count else 0
    
    # Summarize joint errors
    if joint_errors:
        errors_arr = np.array(joint_errors)
        mean_err = np.mean(np.abs(errors_arr), axis=0)
        max_err = np.max(np.abs(errors_arr), axis=0)
    else:
        mean_err = max_err = np.zeros(6)
    
    return {
        "name": solver_name,
        "avg_time_ms": average_time_ms,
        "min_time_ms": min_time_ms,
        "max_time_ms": max_time_ms,
        "success_rate": success_rate,
        "successful": successful_iks,
        "failed": failed_iks,
        "avg_iterations": avg_iterations,
        "mean_joint_error": mean_err,
        "max_joint_error": max_err,
    }


def main():
    num_iterations = args.iterations  # Number of IK calculations to average
    max_ik_time_seconds = 1.0  # Max time to wait for a single IK solution

    # Config files are in the 'configs' directory at the project root
    config_dir_path = os.path.join(project_root, "configs")
    c_path = os.path.join(config_dir_path, "calibration.conf")
    u_path = os.path.join(config_dir_path, "urcontrol.conf.UR5")

    if not (os.path.exists(c_path) and os.path.exists(u_path)):
        logging.error("Config files not found in configs directory under project root.")
        logging.error(f"Expected calibration.conf at: {c_path}")
        logging.error(f"Expected urcontrol.conf.UR5 at: {u_path}")
        sys.exit(1)

    dt, da, dd, dalpha = load_calibration(c_path)
    (a0, d0, alpha0, q_home0, j_dir), tcp_conf = load_urcontrol_config(u_path)

    eff_a, eff_d, eff_alpha = a0 + da, d0 + dd, alpha0 + dalpha
    T_fl_tcp = tcp_transform(tcp_conf)

    # Fix random seed for reproducibility
    np.random.seed(42)
    
    # Run benchmark for numerical IK
    numerical_metrics = run_ik_benchmark(
        "IK Numerical", 
        ik_numerical, 
        num_iterations, 
        eff_a, 
        eff_alpha, 
        eff_d, 
        j_dir, 
        dt, 
        T_fl_tcp, 
        q_home0,
        max_ik_time_seconds
    )
    
    # Reset random seed to use same test cases
    np.random.seed(42)
    
    # Run benchmark for QuIK IK
    quik_metrics = run_ik_benchmark(
        "IK QuIK", 
        ik_quik, 
        num_iterations, 
        eff_a, 
        eff_alpha, 
        eff_d, 
        j_dir, 
        dt, 
        T_fl_tcp, 
        q_home0,
        max_ik_time_seconds
    )
    
    # Print comparison results
    logging.info("\n--- IK Benchmark Comparison ---")
    logging.info(f"Number of iterations: {num_iterations}")
    
    logging.info("\n=== Performance Metrics ===")
    logging.info(f"{'Solver':<15} {'Avg Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15} {'Success Rate (%)':<15} {'Avg Iterations':<15}")
    logging.info(f"{numerical_metrics['name']:<15} {numerical_metrics['avg_time_ms']:<15.4f} {numerical_metrics['min_time_ms']:<15.4f} {numerical_metrics['max_time_ms']:<15.4f} {numerical_metrics['success_rate']:<15.2f} {numerical_metrics['avg_iterations']:<15.2f}")
    logging.info(f"{quik_metrics['name']:<15} {quik_metrics['avg_time_ms']:<15.4f} {quik_metrics['min_time_ms']:<15.4f} {quik_metrics['max_time_ms']:<15.4f} {quik_metrics['success_rate']:<15.2f} {quik_metrics['avg_iterations']:<15.2f}")
    
    # Calculate speedup
    if numerical_metrics['avg_time_ms'] > 0:
        speedup = numerical_metrics['avg_time_ms'] / quik_metrics['avg_time_ms']
        logging.info(f"\nSpeedup (Numerical/QuIK): {speedup:.2f}x")
    
    logging.info("\n=== Accuracy Metrics ===")
    logging.info(f"{'Solver':<15} {'Mean Joint Error (rad)':<50} {'Max Joint Error (rad)':<50}")
    logging.info(f"{numerical_metrics['name']:<15} {str(numerical_metrics['mean_joint_error'].round(6)):<50} {str(numerical_metrics['max_joint_error'].round(6)):<50}")
    logging.info(f"{quik_metrics['name']:<15} {str(quik_metrics['mean_joint_error'].round(6)):<50} {str(quik_metrics['max_joint_error'].round(6)):<50}")
    
    logging.info("\n=== Success/Failure Counts ===")
    logging.info(f"{'Solver':<15} {'Successful':<15} {'Failed':<15}")
    logging.info(f"{numerical_metrics['name']:<15} {numerical_metrics['successful']:<15} {numerical_metrics['failed']:<15}")
    logging.info(f"{quik_metrics['name']:<15} {quik_metrics['successful']:<15} {quik_metrics['failed']:<15}")


if __name__ == "__main__":
    main()
