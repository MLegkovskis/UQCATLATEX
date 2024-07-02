import numpy as np

def function_of_interest(x, a=7, b=0.1):
    return [np.sin(x[0]) + a * np.sin(x[1]) ** 2 + b * x[2] ** 4 * np.sin(x[0])]

# Problem definition
problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'distributions': [
        {'type': 'Uniform', 'params': [-np.pi, np.pi]},  # x1
        {'type': 'Uniform', 'params': [-np.pi, np.pi]},  # x2
        {'type': 'Uniform', 'params': [-np.pi, np.pi]}   # x3
    ]
}

model = function_of_interest
