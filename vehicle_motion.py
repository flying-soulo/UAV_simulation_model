import numpy as np
from body_dynamics import forces_moments_calc
from coordinate_frame_transform import rotation_matrix

# ---------- 6DOF EOM ----------
def six_DOF_motion(vehicle_prop, current_states, throttle, ctrl_srfc_pwm):
    # Extract state variables using array indexing
    u, v, w = current_states[3:6]
    p, q, r = current_states[6:9]
    phi, theta, psi = current_states[9:12]
    
    R_body_to_ned = rotation_matrix(phi, theta, psi)  # Rotation matrix from body to NED frame

    mass = vehicle_prop['m']  # Mass of the vehicle

    # Calculate forces and moments
    Fx, Fy, Fz, Mx, My, Mz = forces_moments_calc(vehicle_prop, current_states, throttle, ctrl_srfc_pwm)
    
    # Acceleration equations
    u_dot = r * v - q * w + Fx / mass
    v_dot = p * w - r * u + Fy / mass
    w_dot = q * u - p * v + Fz / mass

    # Inertial properties of UAV
    j_x = vehicle_prop['Jx']
    j_y = vehicle_prop['Jy']
    j_z = vehicle_prop['Jz']
    j_xz = vehicle_prop['Jxz']

    gamma = j_x * j_z - j_xz**2
    gamma1 = (j_xz * (j_x - j_y + j_z)) / gamma
    gamma2 = (j_z * (j_z - j_y) + j_xz**2) / gamma
    gamma3 = j_z / gamma
    gamma4 = j_xz / gamma
    gamma5 = (j_z - j_x) / j_y
    gamma6 = j_xz / j_y
    gamma7 = (j_x - j_y) * j_x + j_xz**2
    gamma8 = j_x / gamma

    # Moment equations
    p_dot = gamma1 * p * q - gamma2 * q * r             +   gamma3 * Mx + gamma4 * Mz
    q_dot = gamma5 * p * r - gamma6 * (p**2 - r**2)     +   My / j_y
    r_dot = gamma7 * p * q - gamma1 * q * r             +   gamma4 * Mx + gamma8 * Mz

    # Feeding the results
    acc_body = np.array([u_dot, v_dot, w_dot])
    omega_dot = np.array([p_dot, q_dot, r_dot])
    accel_ned = (R_body_to_ned @ acc_body).flatten()


    return np.concatenate((accel_ned, omega_dot))

