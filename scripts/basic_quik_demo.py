import numpy as np
from ur_kinematics_calib.ik import ik_quik
from ur_kinematics_calib.util import load_calibration, load_urcontrol_config
from ur_kinematics_calib.fk import tcp_transform

# Load robot parameters
dt, da, dd, dalpha = load_calibration("configs/calibration.conf")
(a0, d0, alpha0, q_home0, j_dir), tcp_conf = load_urcontrol_config("configs/urcontrol.conf.UR5")

# Calculate effective DH parameters
eff_a = a0 + da
eff_d = d0 + dd
eff_alpha = alpha0 + dalpha

# Calculate TCP transform
T_fl_tcp = tcp_transform(tcp_conf)

# Define target pose (4x4 homogeneous transformation matrix)
T_target = np.eye(4)
T_target[0:3, 3] = [0.6, 0.7, 0.3]  # Position (x, y, z) in meters

# Call IK solver with default parameters (using analytic seed)
q_sol, extra_data = ik_quik(
    eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, 
    use_analytic_seed=True
)

# or
# Call IK solver with default parameters (using previous joint values)
# q_sol, extra_data = ik_quik(
#     eff_a, eff_alpha, eff_d, j_dir, dt, T_fl_tcp, T_target, 
#     use_analytic_seed=False
# )

# Unpack additional information
error, iterations, reason = extra_data

# Print results
print(f"Joint angles (radians): {q_sol}")
print(f"Joint angles (degrees): {np.rad2deg(q_sol)}")
print(f"Error: {np.linalg.norm(error):.6e}")
print(f"Iterations: {iterations}")
print(f"Status: {reason}")