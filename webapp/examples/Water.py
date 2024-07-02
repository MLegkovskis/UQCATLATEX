import numpy as np

# Define the function based on the given symbolic expression
def function_of_interest(inputs):
    rw, r, Tu, Hu, Tl, Hl, L, Kw = inputs
    value = (2 * np.pi * Tu * (Hu - Hl)) / (np.log(r / rw) * (1 + (2 * L * Tu) / (np.log(r / rw) * rw ** 2 * Kw) + Tu / Tl))
    return [value]

# Problem definition
problem = {
    'num_vars': 8,
    'names': ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw"],
    'distributions': [
        {'type': 'Normal', 'params': [0.1, 0.0161812]},   # rw: Normal distribution
        {'type': 'LogNormal', 'params': [7.71, 1.0056]},  # r: LogNormal distribution
        {'type': 'Uniform', 'params': [63070.0, 115600.0]},    # Tu: Uniform distribution
        {'type': 'Uniform', 'params': [990.0, 1110.0]},        # Hu: Uniform distribution
        {'type': 'Uniform', 'params': [63.1, 116.0]},          # Tl: Uniform distribution
        {'type': 'Uniform', 'params': [700.0, 820.0]},         # Hl: Uniform distribution
        {'type': 'Uniform', 'params': [1120.0, 1680.0]},       # L: Uniform distribution
        {'type': 'Uniform', 'params': [9855.0, 12045.0]}       # Kw: Uniform distribution
    ]
}

model = function_of_interest
