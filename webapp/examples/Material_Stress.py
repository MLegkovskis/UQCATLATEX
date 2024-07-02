import numpy as np

# Define your function here
def function_of_interest(inputs):
    gamma, phi, Rs, G, M = inputs
    b = 2.54e-9  # Constant value of b
    sigma_c_mpa = abs(((M * gamma) / (2.0 * b)) * (np.sqrt((8.0 * gamma * phi * Rs) / (np.pi * G * pow(b, 2))) - phi) / 1e6)
    return [sigma_c_mpa]

# Problem definition
problem = {
    'num_vars': 5,
    'names': ['gamma', 'phi', 'Rs', 'G', 'M'],
    'distributions': [
        {'type': 'Uniform', 'params': [0.15, 0.25]},        # gamma
        {'type': 'Uniform', 'params': [0.30, 0.45]},        # phi
        {'type': 'Uniform', 'params': [1e-8, 3e-8]},        # Rs
        {'type': 'Uniform', 'params': [6e10, 8e10]},        # G
        {'type': 'Normal', 'params': [3.05, 0.15]}          # M
    ]
}

model = function_of_interest
