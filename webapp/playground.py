from flask import Flask, request, render_template, jsonify, send_file
import importlib.util
import sys
import os
import numpy as np
import pandas as pd
import openturns as ot
import openturns.viewer as otv
import matplotlib.pyplot as plt
import shutil
import requests
import traceback
import matplotlib.ticker as ticker
import shutil



app = Flask(__name__)

# Constants for directories
OUTPUT_DIR = "results"
OUTPUT_VISUAL_DIR = "results_visual"
OUTPUT_API_DIR = "results_api"
FIGURES_DIR = "figures"


shutil.rmtree(OUTPUT_DIR)
shutil.rmtree(OUTPUT_VISUAL_DIR)
shutil.rmtree(OUTPUT_API_DIR)
shutil.rmtree(FIGURES_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_VISUAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_API_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)



file_path = 'user_model.py'
# Load the user model
MODULE_NAME = "UserModel"
spec = importlib.util.spec_from_file_location(MODULE_NAME, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)

function_of_interest, problem = module.model, module.problem

# Log the problem definition
app.logger.info(f"Problem definition: {problem}")

# Create distributions
distributions = ot.DistributionCollection()
for dist_info in problem['distributions']:
    dist_type = dist_info['type']
    params = dist_info['params']
    if dist_type == 'Uniform':
        distributions.add(ot.Uniform(*params))
    elif dist_type == 'Normal':
        distributions.add(ot.Normal(*params))
    elif dist_type == 'LogNormalMuSigma':
        distributions.add(ot.ParametrizedDistribution(ot.LogNormalMuSigma(*params)))
    elif dist_type == 'LogNormal':
        distributions.add(ot.LogNormal(*params))
    elif dist_type == 'Beta':
        distributions.add(ot.Beta(*params))
    elif dist_type == 'Gumbel':
        distributions.add(ot.Gumbel(*params))
    elif dist_type == 'Triangular':
        distributions.add(ot.Triangular(*params))
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

distribution = ot.ComposedDistribution(distributions)
input_names = problem['names']
print(input_names)
# Define the OpenTURNS model
ot_model = ot.PythonFunction(problem['num_vars'], 1, function_of_interest)

# Draw the function
n = 10000
sampleX = distribution.getSample(n)
sampleY = ot_model(sampleX)

# Save data used in plotXvsY to CSV
sampleX.exportToCSVFile(os.path.join(OUTPUT_DIR, "X.csv"), ",")
sampleY.exportToCSVFile(os.path.join(OUTPUT_DIR, "Y.csv"), ",")

X = pd.read_csv(os.path.join(OUTPUT_DIR, 'X.csv'))
Y = pd.read_csv(os.path.join(OUTPUT_DIR, 'Y.csv'))
X.columns = problem['names']

grid_df = pd.concat([Y, X], axis=1)
grid_df.to_csv(os.path.join(OUTPUT_DIR, "grid.csv"), index=False)

# Estimate the Sobol' indices
size = 1000
sie = ot.SobolIndicesExperiment(distribution, size)
inputDesign = sie.generate()
inputDesign.setDescription(input_names)
outputDesign = ot_model(inputDesign)

sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)

graph = sensitivityAnalysis.draw()
view = otv.View(graph)
plt.savefig(os.path.join(OUTPUT_VISUAL_DIR, "sobol_indices.png"))

# Create DataFrames for Sobol indices
rows = str(sensitivityAnalysis.getFirstOrderIndicesInterval()).split('\n')
data = [tuple(map(float, row.strip('[]').split(','))) for row in rows]
df = pd.DataFrame(data, columns=['Upper Bound', 'Lower Bound'])
new_df = pd.DataFrame({'Sobol Index': list(map(float, str(sensitivityAnalysis.getFirstOrderIndices()).strip('[]').split(',')))})
first_order_df = pd.concat([pd.DataFrame({'Inputs': input_names}), new_df, df], axis=1)

rows = str(sensitivityAnalysis.getTotalOrderIndicesInterval()).split('\n')
data = [tuple(map(float, row.strip('[]').split(','))) for row in rows]
df = pd.DataFrame(data, columns=['Upper Bound', 'Lower Bound'])
new_df = pd.DataFrame({'Sobol Index': list(map(float, str(sensitivityAnalysis.getTotalOrderIndices()).strip('[]').split(',')))})
total_order_df = pd.concat([pd.DataFrame({'Inputs': input_names}), new_df, df], axis=1)


expectation_convergence_df = pd.read_csv(os.path.join(OUTPUT_API_DIR, 'expectation_convergence.csv'))
expectation_convergence_csv = expectation_convergence_df.iloc[::500].to_csv(index=False)

print(expectation_convergence_csv)


