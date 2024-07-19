import numpy as np

# Define the Wing weight function
def wing_weight_function(inputs):
    S_w, W_fw, A, Lambda, q, ell, t_c, N_z, W_dg, W_p = inputs
    g = (0.036 * S_w**0.758 * W_fw**0.0035 * (A / np.cos(np.radians(Lambda))**2)**0.6 *
         q**0.006 * ell**0.04 * (100 * t_c / np.cos(np.radians(Lambda)))**-0.3 *
         (N_z * W_dg)**0.49) + S_w * W_p
    return [g]

# Problem definition
problem = {
    'num_vars': 10,
    'names': ["S_w", "W_fw", "A", "Lambda", "q", "ell", "t_c", "N_z", "W_dg", "W_p"],
    'distributions': [
        {'type': 'Uniform', 'params': [150.0, 200.0]},    # S_w: Uniform distribution (ft^2)
        {'type': 'Uniform', 'params': [220.0, 300.0]},    # W_fw: Uniform distribution (lb)
        {'type': 'Uniform', 'params': [6.0, 10.0]},       # A: Uniform distribution (-)
        {'type': 'Uniform', 'params': [-10.0, 10.0]},     # Lambda: Uniform distribution (deg)
        {'type': 'Uniform', 'params': [16.0, 45.0]},      # q: Uniform distribution (lb/ft^2)
        {'type': 'Uniform', 'params': [0.5, 1.0]},        # ell: Uniform distribution (-)
        {'type': 'Uniform', 'params': [0.08, 0.18]},      # t_c: Uniform distribution (-)
        {'type': 'Uniform', 'params': [2.5, 6.0]},        # N_z: Uniform distribution (-)
        {'type': 'Uniform', 'params': [1700.0, 2500.0]},  # W_dg: Uniform distribution (lb)
        {'type': 'Uniform', 'params': [0.025, 0.08]}      # W_p: Uniform distribution (lb/ft^2)
    ]
}

model = wing_weight_function
