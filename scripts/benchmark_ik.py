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
    from ur_kinematics_calib.ik import ik_numerical

    logging.info("Successfully imported UR kinematics modules")
except ImportError:
    logging.error("Error: Could not import from ur_kinematics_calib.")
    logging.error(
        "Make sure 'ur_kinematics_calib' is in the project root and contains '__init__.py'."
    )
    logging.error(f"Project root added to sys.path: {project_root}")
    sys.exit(1)


def main():
    num_iterations = 500  # Number of IK calculations to average
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

    logging.info(f"Starting IK benchmark with {num_iterations} iterations...\n")
    # print("Effective D-H (a, d in m; alpha in rad):")
    # print(f"  a_eff:     {eff_a.round(6).tolist()}")
    # print(f"  d_eff:     {eff_d.round(6).tolist()}")
    # print(f"  alpha_eff: {eff_alpha.round(6).tolist()}\n")

    ik_times = []
    successful_iks = 0
    failed_iks = 0
    joint_errors = []

    for i in tqdm(range(num_iterations), desc="IK iterations"):
        # 1. Generate random joint angles to create a reachable target pose
        random_joints_rad_for_target = np.random.uniform(-np.pi, np.pi, 6)
        dh_thetas_for_target = random_joints_rad_for_target + dt

        # 2. Calculate FK to get the target TCP pose
        T_base_fl_target = fk_to_flange(
            eff_a, eff_alpha, eff_d, j_dir, dh_thetas_for_target
        )
        T_target_tcp = T_base_fl_target @ T_fl_tcp

        # 3. Use q_home0 as the initial guess for IK
        q_init_ik = (
            q_home0.copy()
        )  # Use a copy to avoid modification if ik_numerical changes it

        start_time = time.perf_counter()
        try:
            # ik_numerical might take a while, consider a timeout mechanism if it hangs
            # For simplicity, we'll rely on least_squares' own convergence criteria
            q_sol, ik_res = ik_numerical(
                eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target_tcp, q_init_ik
            )
            end_time = time.perf_counter()

            duration = end_time - start_time
            ik_times.append(duration)

            if (
                ik_res.success and duration < max_ik_time_seconds
            ):  # Check if solver reported success
                successful_iks += 1
            else:
                failed_iks += 1
                if not ik_res.success:
                    logging.warning(
                        f"Iteration {i + 1}: IK failed to converge. Cost: {ik_res.cost:.2e}"
                    )
                if duration >= max_ik_time_seconds:
                    logging.warning(
                        f"Iteration {i + 1}: IK timed out ({duration:.2f}s)."
                    )

        except Exception as e:
            end_time = time.perf_counter()  # still record time up to failure
            ik_times.append(end_time - start_time)
            failed_iks += 1
            logging.warning(f"Iteration {i + 1}: IK solver raised an exception: {e}")

        # Compare IK solution to original random joints
        error = q_sol - random_joints_rad_for_target
        joint_errors.append(error)

        # if (i + 1) % (num_iterations // 10 if num_iterations >=10 else 1) == 0:
        #     print(f"Completed {i + 1}/{num_iterations} iterations... (Successes: {successful_iks}, Failures: {failed_iks})")

    if ik_times:  # Ensure ik_times is not empty
        average_time_ms = (sum(ik_times) / len(ik_times)) * 1000
        min_time_ms = min(ik_times) * 1000
        max_time_ms = max(ik_times) * 1000
        success_rate = (successful_iks / num_iterations) * 100
    else:
        average_time_ms = min_time_ms = max_time_ms = float("nan")
        success_rate = 0.0

    logging.info("\n--- IK Benchmark Results ---")
    logging.info(f"Number of iterations: {num_iterations}")
    if ik_times:
        logging.info(f"Average IK calculation time: {average_time_ms:.4f} ms")
        logging.info(f"Min IK calculation time: {min_time_ms:.4f} ms")
        logging.info(f"Max IK calculation time: {max_time_ms:.4f} ms")
    else:
        logging.info("No IK calculations were successfully timed.")
    logging.info(
        f"Successful IK solutions: {successful_iks}/{num_iterations} ({success_rate:.2f}%)"
    )
    logging.info(f"Failed IK solutions (no convergence): {failed_iks}")

    # Summarize joint errors
    if joint_errors:
        errors_arr = np.array(joint_errors)
        mean_err = np.mean(np.abs(errors_arr), axis=0)
        max_err = np.max(np.abs(errors_arr), axis=0)
        logging.info(f"Joint error mean per joint (rad): {mean_err.round(6).tolist()}")
        logging.info(f"Joint error max per joint (rad): {max_err.round(6).tolist()}")


if __name__ == "__main__":
    main()
