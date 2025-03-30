import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from vehicle_properties import get_aerosnode_properties
from vehicle_motion import six_DOF_motion
from math_utils import wrap

def run_simulation(T_end=10.0, dt=0.1):
    t = np.arange(0, T_end, dt)  # Time array
    
    # vehicle properties
    vehicle_prop = get_aerosnode_properties()

    throttle = (0.0, 0.0, 0.0, 0.0, 0.0)# Throttle values for motors and control surfaces set to a small value
    ctrl_srfc_pwm = np.full(3, 0)  # Control surface PWM values set to a small value

    # Dictionary to store simulation results with variable names as keys
    simulation_data = {
        "time": [],
        "x": [],
        "y": [],
        "z": [],
        "u": [],
        "v": [],
        "w": [],
        "phi": [],
        "theta": [],
        "psi": [],
        "p": [],
        "q": [],
        "r": [],
    }

    # Simulation loop
    current_state = np.zeros(12)  # Initialize current state variables
    for time in t:
        ''' represent the states of the vehicle. in the order of: (x, y , z, u, v, w, phi, theta, psi, p, q, r) '''
        next_state = np.zeros(12)  # Next state variables

        derivatives = six_DOF_motion(vehicle_prop, current_state, throttle, ctrl_srfc_pwm)

        # Update velocity states
        next_state[3] = current_state[3] + derivatives[0] * dt  # u (velocity in x-direction)
        next_state[4] = current_state[4] + derivatives[1] * dt  # v (velocity in y-direction)
        next_state[5] = current_state[5] + derivatives[2] * dt  # w (velocity in z-direction)

        # Update angular rate states
        next_state[9] = current_state[9] + derivatives[3] * dt  # p (angular velocity about x-axis)
        next_state[10] = current_state[10] + derivatives[4] * dt  # q (angular velocity about y-axis)
        next_state[11] = current_state[11] + derivatives[5] * dt  # r (angular velocity about z-axis)

        # Update and wrap position states
        next_state[0] = wrap(current_state[0] + current_state[3] * dt, -np.inf, np.inf)  # x (position in x-direction)
        next_state[1] = wrap(current_state[1] + current_state[4] * dt, -np.inf, np.inf)  # y (position in y-direction)
        next_state[2] = wrap(current_state[2] + current_state[5] * dt, 0, np.inf)  # z (position in z-direction, constrained to ground level)

        # Wrap angles between -180 and 180 degrees
        next_state[6] = (current_state[6] + current_state[9] * dt + 180) % 360 - 180  # phi (roll angle)
        next_state[7] = (current_state[7] + current_state[10] * dt + 180) % 360 - 180  # theta (pitch angle)
        next_state[8] = (current_state[8] + current_state[11] * dt + 180) % 360 - 180  # psi (yaw angle)

        # Append data to the dictionary
        simulation_data["time"].append(time)
        for i, key in enumerate(
            ["x", "y", "z", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r"]
        ):
            simulation_data[key].append(current_state[i])

        # Update current state for the next iteration
        current_state = next_state.copy()

    # Convert simulation data to DataFrame
    df = pd.DataFrame(simulation_data)

    # Save to CSV (overwrite mode)
    df.to_csv("simulation_log.csv", index=False)

    return t, df


def plot_states(t, states):
    labels = [
        "x (m)",       # Position in x-direction
        "y (m)",       # Position in y-direction
        "z (m)",       # Position in z-direction
        "u (m/s)",     # Velocity in x-direction
        "v (m/s)",     # Velocity in y-direction
        "w (m/s)",     # Velocity in z-direction
        "phi (rad)",   # Roll angle
        "theta (rad)", # Pitch angle
        "psi (rad)",   # Yaw angle
        "p (rad/s)",   # Angular velocity about x-axis
        "q (rad/s)",   # Angular velocity about y-axis
        "r (rad/s)",   # Angular velocity about z-axis
    ]
    plt.figure(figsize=(15, 10))
    for i in range(12):
        plt.subplot(4, 3, i + 1)
        plt.plot(t, states[:, i])
        plt.xlabel("Time (s)")
        plt.ylabel(labels[i])
        plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    t, df = run_simulation()
    states = df.iloc[:, 1:].values  # Extract state data from DataFrame
    plot_states(t, states)
