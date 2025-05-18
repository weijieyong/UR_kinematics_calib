import numpy as np
import time
from typing import Tuple, List
import quik_bind as quikpy
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RobotConfig:
    """Configuration for UR5 manipulator DH parameters and joint types."""

    DH_PARAMS = np.array(
        [
            [
                0.0 + 0.000143412481775751,
                1.570796327 + 0.000261142842141071,
                0.1625 + 0.000229096222625319,
                0.0 + 2.6321312115995532e-08,
            ],
            [
                -0.425 + 0.0111645466532434,
                0.0 + 0.0004789150738772391,
                0.0 - 204.07710540653042,
                0.0 - 0.23192374554198558,
            ],
            [
                -0.3922 - 7.29306429188004e-05,
                0.0 + 0.007380884399427977,
                0.0 + 203.39232363068487,
                0.0 + 6.502222138633594,
            ],
            [
                0.0 + 0.00019069275578740042,
                1.570796327 + 0.000252176196238407,
                0.1333 + 0.685273448748188,
                0.0 + 0.012880087265360121,
            ],
            [
                0.0 + 1.5310755735655134e-06,
                -1.570796327 + 0.001845349173593247,
                0.0997 - 0.00012524970042109,
                0.0 - 6.757964824549712e-08,
            ],
            [0.0, 0.0, 0.0996 - 0.00013082748012806233, 0.0 - 7.098998536340939e-07],
        ],
        dtype=np.float64,
    )

    LINK_TYPES = np.array([False] * 6, dtype=bool)
    NUM_SAMPLES = 10
    MAX_ITER = 100
    ERROR_THRESHOLD = 1e-2


def initialize_robot(dh_params: np.ndarray, link_types: np.ndarray) -> None:
    """Initialize the robot with DH parameters and link types."""
    try:
        quikpy.init_robot(dh_params, link_types)
        logger.info("Robot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize robot: {e}")
        raise


def generate_random_configurations(dof: int, num_samples: int) -> np.ndarray:
    """Generate random joint configurations."""
    return np.random.uniform(-np.pi, np.pi, size=(dof, num_samples))


def compute_forward_kinematics(
    joint_configs: np.ndarray,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Compute forward kinematics for given joint configurations."""
    num_samples = joint_configs.shape[1]
    Tn = np.zeros((4 * num_samples, 4), dtype=np.float64)
    Ts = []

    for i in range(num_samples):
        try:
            T = quikpy.fkn(joint_configs[:, i])
            Ts.append(T)
            Tn[i * 4 : (i + 1) * 4, :] = T
        except Exception as e:
            logger.error(f"FK computation failed for sample {i+1}: {e}")
            raise

    return Tn, Ts


def solve_inverse_kinematics(
    poses: List[np.ndarray], initial_guesses: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str], float]:
    """Solve inverse kinematics for given poses."""
    num_samples = len(poses)
    # dof = initial_guesses.shape[0]  # commented out not used
    Q_star = np.zeros_like(initial_guesses)
    E_star = np.zeros((6, num_samples))
    iters = []
    reasons = []

    start_time = time.perf_counter()
    for i in range(num_samples):
        try:
            q_sol, e_sol, itr, reason = quikpy.ik(poses[i], initial_guesses[:, i])
            Q_star[:, i] = q_sol
            E_star[:, i] = e_sol
            iters.append(itr)
            reasons.append(reason)
        except Exception as e:
            logger.error(f"IK computation failed for sample {i+1}: {e}")
            raise

    elapsed = (time.perf_counter() - start_time) * 1e6
    return Q_star, E_star, iters, reasons, elapsed


def print_results(
    Q: np.ndarray,
    Q0: np.ndarray,
    Q_star: np.ndarray,
    E_star: np.ndarray,
    iters: List[int],
    reasons: List[str],
    elapsed: float,
    Ts: List[np.ndarray],
) -> None:
    """Print detailed results of IK computation."""
    num_samples = Q.shape[1]

    logger.info("=== Random Joint Angles Generated ===")
    for i in range(num_samples):
        angles_deg = Q[:, i] * 180.0 / np.pi
        logger.debug(f"Config {i+1}: {angles_deg}")

    logger.info("=== Random Initial Guesses ===")
    for i in range(num_samples):
        angles_deg = Q0[:, i] * 180.0 / np.pi
        logger.debug(f"Initial guess {i+1}: {angles_deg}")

    logger.info("=== Forward Kinematics Poses ===")
    for i, T in enumerate(Ts):
        pos = T[:3, 3]
        logger.debug(f"Pose {i+1}: Position = {pos}")

    logger.info("=== IK Results ===")
    logger.info(f"Iterations: {iters}")
    logger.info(f"Reasons: {reasons}")
    logger.info(f"Time elapsed (total) [µs]: {elapsed:.2f}")
    logger.info(f"Time per sample [µs]: {elapsed/num_samples:.2f}")

    logger.info("=== Joint Angle Comparison ===")
    logger.debug("Starting joint angles (Q0):\n%s", Q0)
    logger.debug("True joint angles (Q):\n%s", Q)
    logger.debug("Final joint angles from IK (Q_star):\n%s", Q_star)

    logger.info("=== Final Normed Pose Errors ===")
    error_norms = np.linalg.norm(E_star, axis=0)
    for i, error in enumerate(error_norms, 1):
        marker = " ***" if error > RobotConfig.ERROR_THRESHOLD else ""
        logger.info(f"Sample {i:2d}: {error:.2e}{marker}")


def main() -> None:
    """Main function to run IK solver demonstration."""
    try:
        # Initialize robot
        initialize_robot(RobotConfig.DH_PARAMS, RobotConfig.LINK_TYPES)
        dof = RobotConfig.DH_PARAMS.shape[0]

        # Generate random configurations
        Q = generate_random_configurations(dof, RobotConfig.NUM_SAMPLES)
        Q0 = generate_random_configurations(dof, RobotConfig.NUM_SAMPLES)

        # Compute forward kinematics
        Tn, Ts = compute_forward_kinematics(Q)

        # Solve inverse kinematics
        Q_star, E_star, iters, reasons, elapsed = solve_inverse_kinematics(Ts, Q0)

        # Print results
        print_results(Q, Q0, Q_star, E_star, iters, reasons, elapsed, Ts)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
