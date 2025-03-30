# This file contains the vehicle properties for the Aerosonde UAV.

Aerosonde_vehicle = {
    'm': 11.0,  # Mass in kg
    'Jx': 0.824,  # Moment of inertia around x-axis in kg-m^2
    'Jy': 1.135,  # Moment of inertia around y-axis in kg-m^2
    'Jz': 1.759,  # Moment of inertia around z-axis in kg-m^2
    'Jxz': 0.120,  # Product of inertia in kg-m^2
    'S': 0.55,  # Wing area in m^2
    'b': 2.9,  # Wing span in m
    'c': 0.19,  # Mean aerodynamic chord in m
    'rho': 1.268,  # Air density in kg/m^3
    'e': 0.9,  # Oswald efficiency factor
    
    #propulsion coefficients
    'Vmax': 44.4,  # Maximum voltage in V
    'Dprop': 0.508,  # Propeller diameter in m
    'KV': 0.0659,  # Motor velocity constant in V-s/rad
    'KQ': 0.0659,  # Motor torque constant in N-m
    'Rmotor': 0.042,  # Motor resistance in ohms
    'i0': 1.5,  # No-load current in A
    'CQ2': -0.01664,  # Propeller torque coefficient term 2
    'CQ1': 0.004970,  # Propeller torque coefficient term 1
    'CQ0': 0.005230,  # Propeller torque coefficient term 0
    'CT2': -0.1079,  # Propeller thrust coefficient term 2
    'CT1': -0.06044,  # Propeller thrust coefficient term 1
    'CT0': 0.09357,   # Propeller thrust coefficient term 0
    
    # Longitudinal Coefficients
    'CL0': 0.23,
    'CD0': 0.043,
    'Cm0': 0.0135,
    'CL_alpha': 5.61,
    'CD_alpha': 0.030,
    'Cm_alpha': -2.74,
    'CLq': 7.95,
    'CDq': 0,
    'Cmq': -38.21,
    'CL_delta_e': 0.13,
    'CD_delta_e': 0.0135,
    'Cm_delta_e': -0.99,
    'M': 50,
    'alpha0': 0.47,
    'CDp': 0.043,

    # Lateral Coefficients
    'CY0': 0,
    'Cl0': 0,
    'Cn0': 0,
    'CY_beta': -0.83,
    'Cl_beta': -0.13,
    'Cn_beta': 0.073,
    'CYp': 0,
    'Clp': -0.51,
    'Cnp': -0.069,
    'CYr': 0,
    'Clr': 0.25,
    'Cnr': -0.095,
    'CY_delta_a': 0.075,
    'Cl_delta_a': 0.17,
    'Cn_delta_a': -0.011,
    'CY_delta_r': 0.19,
    'Cl_delta_r': 0.0024,
    'Cn_delta_r': -0.069,
    
    #Quadcopter properties
    'arm_length': 0.5,  # Arm length in meters
    'toruq_motor': 0.1,  # Torque of the motor in N-m
    'max_thrust': 10.0,  # Maximum thrust in N
}

def get_aerosnode_properties():
    return Aerosonde_vehicle.copy()
