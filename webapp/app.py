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
from prompt import get_prompt


app = Flask(__name__)

# Constants for directories
OUTPUT_DIR = "results"
OUTPUT_VISUAL_DIR = "results_visual"
OUTPUT_API_DIR = "results_api"
FIGURES_DIR = "figures"
EXAMPLES_DIR = "examples"

# Clear and create directories
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
shutil.rmtree(OUTPUT_VISUAL_DIR, ignore_errors=True)
shutil.rmtree(OUTPUT_API_DIR, ignore_errors=True)
shutil.rmtree(FIGURES_DIR, ignore_errors=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_VISUAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_API_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

@app.route('/')
def index():
    # List all example files
    examples = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith('.py')]
    return render_template('index.html', examples=examples)

@app.route('/load_example', methods=['GET'])
def load_example():
    example = request.args.get('example')
    if example:
        with open(os.path.join(EXAMPLES_DIR, example), 'r') as file:
            content = file.read()
        # Remove the last line "model = function_of_interest"
        content = '\n'.join(content.split('\n')[:-1])
        return content
    return '', 404

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Save the uploaded file
        user_model_code = request.form['user_model_code']
        with open('user_model.py', 'w') as file:
            file.write(user_model_code)

        # Log the user model code
        app.logger.info(f"User model code:\n{user_model_code}")

        # Run the analysis script
        run_analysis_script('user_model.py')
        
        # Make API call to ChatGPT
        latex_code = call_chatgpt_analysis(user_model_code)

        # Generate PDF from LaTeX code
        pdf_path = generate_pdf_from_latex(latex_code)

        # Return the LaTeX as a response
        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def run_analysis_script(file_path):
    app.logger.info(f"Running analysis script with {file_path}")
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

    # Save Sobol indices to CSV
    first_order_df.to_csv(os.path.join(OUTPUT_DIR, "first_order_sobol_indices.csv"), index=False)
    total_order_df.to_csv(os.path.join(OUTPUT_DIR, "total_order_sobol_indices.csv"), index=False)

    # Define the input distribution
    input_vector = ot.RandomVector(distribution)

    # The output vector is a CompositeRandomVector
    output_vector = ot.CompositeRandomVector(ot_model, input_vector)

    # Define the algorithm for expectation convergence
    algo = ot.ExpectationSimulationAlgorithm(output_vector)
    algo.setMaximumOuterSampling(1000)
    algo.setBlockSize(1)
    algo.setCoefficientOfVariationCriterionType("NONE")

    # Run the algorithm and store the result
    algo.run()
    result = algo.getResult()

    # Draw the convergence history and save the convergence data to a CSV file
    graphConvergence = algo.drawExpectationConvergence()
    data = graphConvergence.getDrawable(0).getData()
    sample_sizes = data[:, 0]
    mean_estimates = data[:, 1]

    # Compute standard deviations for the mean estimates
    standard_deviations = result.getStandardDeviation()

    # Calculate confidence intervals
    z_value = 1.96  # For a 95% confidence interval
    lower_bounds = mean_estimates - z_value * standard_deviations
    upper_bounds = mean_estimates + z_value * standard_deviations

    df = pd.DataFrame({
        "Sample Size": [point[0] for point in sample_sizes],
        "Mean Estimate": [point[0] for point in mean_estimates],
        "Lower Bound": [point[0] for point in lower_bounds],
        "Upper Bound": [point[0] for point in upper_bounds]
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "expectation_convergence.csv"), index=False)

    # Perform correlation analysis and save results to CSV
    corr_analysis = ot.CorrelationAnalysis(sampleX, sampleY)

    methods = {
        "PCC": corr_analysis.computePCC,
        "PRCC": corr_analysis.computePRCC,
        "SRC": corr_analysis.computeSRC,
        "SRRC": corr_analysis.computeSRRC,
        "Pearson": corr_analysis.computePearsonCorrelation,
        "Spearman": corr_analysis.computeSpearmanCorrelation,
    }

    for method, func in methods.items():
        indices = func()
        data = {
            'Variable': f"[{','.join(input_names)}]",
            'Correlation_Coefficient': list(indices)
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{method}_coefficients.csv"), index=False)

    # Combine CSV files for correlation coefficients
    combined_df = None
    for file in os.listdir(OUTPUT_DIR):
        if file.endswith('_coefficients.csv'):
            method = file.replace('_coefficients.csv', '')
            df = pd.read_csv(os.path.join(OUTPUT_DIR, file))
            df = df.rename(columns={'Correlation_Coefficient': method})
            if combined_df is None:
                combined_df = df
            else:
                combined_df[method] = df[method]
    combined_df.to_csv(os.path.join(OUTPUT_DIR, 'combined_coefficients.csv'), index=False)

    # Transform combined coefficients CSV for API consumption
    combined_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'combined_coefficients.csv'))
    variables = combined_df['Variable'][0].strip('[]').split(',')
    combined_df['Variable'] = variables
    combined_df.to_csv(os.path.join(OUTPUT_API_DIR, 'combined_coefficients_transformed.csv'), index=False)

    # Copy necessary files to the API results directory
    files_to_copy = [
        os.path.join(OUTPUT_DIR, "expectation_convergence.csv"),
        os.path.join(OUTPUT_DIR, "first_order_sobol_indices.csv"),
        os.path.join(OUTPUT_DIR, "total_order_sobol_indices.csv")
    ]
    for file in files_to_copy:
        destination_file = os.path.join(OUTPUT_API_DIR, os.path.basename(file))
        shutil.copyfile(file, destination_file)

    # Generate plots
    generate_plots()

def generate_plots():
    # Plot correlation coefficients
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'combined_coefficients.csv'))
    xticks = df['Variable'][0].strip('[]').replace("'", "").split(',')
    df['Variable'] = df.index
    df.set_index('Variable', inplace=True)
    ax = df.plot(kind='bar', figsize=(12, 8))
    plt.xlabel('Variable Group Index')
    plt.ylabel('Correlation Coefficient')
    plt.title('Comparison of Correlation Coefficients Across Variables')
    plt.legend(title='Method')
    ax.set_xticklabels(xticks, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VISUAL_DIR, "correlation_coefficients.png"))
    plt.close()

    # Plot expectation convergence
    data = pd.read_csv(os.path.join(OUTPUT_DIR, "expectation_convergence.csv"))
    plt.figure(figsize=(10, 6))
    plt.plot(data['Sample Size'], data['Mean Estimate'], label='Mean Estimate', color='blue')
    plt.fill_between(data['Sample Size'], data['Lower Bound'], data['Upper Bound'], color='blue', alpha=0.2, label='95% Confidence Interval')
    plt.title('Mean Estimate Convergence with Confidence Intervals')
    plt.xlabel('Sample Size')
    plt.ylabel('Mean Estimate')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_VISUAL_DIR, "mean_estimate_convergence_plot.png"))
    plt.close()

    # Plot grid data
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'grid.csv'))
    y = df['y0']
    X = df.drop(columns=['y0'])
    X_names = X.columns
    dimX = X.shape[1]
    fig, axes = plt.subplots(1, dimX, figsize=(15, 5))
    for j in range(dimX):
        ax = axes[j] if dimX > 1 else axes
        ax.scatter(X.iloc[:, j], y, alpha=0.5, s=5)
        ax.set_xlabel(X_names[j], fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=90)
        if j == 0:
            ax.set_ylabel('y0')
        else:
            ax.set_ylabel("")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VISUAL_DIR, "grid_plot.png"))
    plt.close()

    # Copy all generated figures to the FIGURES_DIR
    figures_to_copy = [
        "correlation_coefficients.png",
        "grid_plot.png",
        "mean_estimate_convergence_plot.png",
        "sobol_indices.png"
    ]
    for figure in figures_to_copy:
        shutil.copyfile(os.path.join(OUTPUT_VISUAL_DIR, figure), os.path.join(FIGURES_DIR, figure))


def call_chatgpt_analysis(user_model_code):
    try:
        # Read CSV files
        def read_csv(file_path):
            with open(file_path, 'r') as file:
                return file.read()

        combined_coefficients_csv = read_csv(os.path.join(OUTPUT_API_DIR, 'combined_coefficients_transformed.csv'))
        expectation_convergence_df = pd.read_csv(os.path.join(OUTPUT_API_DIR, 'expectation_convergence.csv'))
        expectation_convergence_csv = expectation_convergence_df.iloc[::500].to_csv(index=False)
        first_order_sobol_csv = read_csv(os.path.join(OUTPUT_API_DIR, 'first_order_sobol_indices.csv'))
        total_order_sobol_csv = read_csv(os.path.join(OUTPUT_API_DIR, 'total_order_sobol_indices.csv'))

        # Get the prompt from prompt.py
        prompt = get_prompt(user_model_code, combined_coefficients_csv, expectation_convergence_csv, first_order_sobol_csv, total_order_sobol_csv)

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are an expert consultant in Uncertainty Quantification who prepares UQ reports for Fortune 500 companies"},
                {"role": "user", "content": prompt}
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        app.logger.error(f"Error in call_chatgpt_analysis: {e}")
        app.logger.error(traceback.format_exc())
        raise




def call_chatgpt_analysis(user_model_code):
    try:
        # Read CSV files
        def read_csv(file_path):
            with open(file_path, 'r') as file:
                return file.read()

        combined_coefficients_csv = read_csv(os.path.join(OUTPUT_API_DIR, 'combined_coefficients_transformed.csv'))
        expectation_convergence_df = pd.read_csv(os.path.join(OUTPUT_API_DIR, 'expectation_convergence.csv'))
        expectation_convergence_csv = expectation_convergence_df.iloc[::500].to_csv(index=False)
        first_order_sobol_csv = read_csv(os.path.join(OUTPUT_API_DIR, 'first_order_sobol_indices.csv'))
        total_order_sobol_csv = read_csv(os.path.join(OUTPUT_API_DIR, 'total_order_sobol_indices.csv'))

        # Define the payload for the API call with full CSV data
        prompt = f"""
        I have a numpy-based model and corresponding CSV data for Uncertainty Quantification (UQ) and Sensitivity Analysis (SA). Please perform the following analysis and generate a LaTeX report. The report should include a model overview, expectation convergence analysis, sensitivity analysis, key findings, conclusions, and insights for decision making. The figures referenced will be saved in the `figures` directory with the names specified.

        ### Here is the model of interest:
        ```python
        {user_model_code}
        ```

        ### CSV Data Format
        The CSV files will be in the following format:
        - `combined_coefficients_transformed.csv`: Contains the correlation coefficients with columns `Variable`, `PCC`, `Pearson`, `PRCC`, `Spearman`, `SRC`, `SRRC`.
        ```csv
        {combined_coefficients_csv}
        ```
        - `expectation_convergence.csv`: Contains the mean estimate convergence data with columns `Sample Size`, `Mean Estimate`, `Lower Bound`, `Upper Bound`.
        ```csv
        {expectation_convergence_csv}
        ```
        - `first_order_sobol_indices.csv`: Contains the first-order Sobol indices with columns `Sobol Index`, `Upper Bound`, `Lower Bound`.
        ```csv
        {first_order_sobol_csv}
        ```
        - `total_order_sobol_indices.csv`: Contains the total-order Sobol indices with columns `Sobol Index`, `Upper Bound`, `Lower Bound`.
        ```csv
        {total_order_sobol_csv}
        ```

        ### Expected Report Structure:

        1. **Model Overview and Uncertain inputs in question:**
           - Brief description of the model and input parameters in LaTeX format - never simply reprint the model as python code, always convert the numpy-based model to the latex format - both the actual model as latex code!
           - State all the uncertainties in question and the associated probability distriubtions AND the associated numbers!
           - Try to understand what the model is in terms of the output and its physical meaning even if the user provided pure python code without comments
           - Try to deduce units of all the variables and use them for better clarity
           - In this section you must make reference to grid_plot.png

        2. **Expectation Convergence Analysis:**
           - Summary and interpretation of the mean estimate convergence data (using `expectation_convergence.csv`). Explain the likely operating range of the model under the specified uncertainty.
           - In this section you must make reference to mean_estimate_convergence_plot.png

        3. **Sensitivity Analysis:**
           - Detailed analysis of correlation coefficients (using `combined_coefficients_transformed.csv`). Ensure you discuss specific, such as why is PRCC coefficient higher than other coefficients as an example only as it not always the case just giving you an idea of discussion avenues! Really understand the data.
           - Detailed analysis of Sobol coefficients (using `first_order_sobol_indices.csv` and `total_order_sobol_indices.csv`) - link it to the correlation coefficients and see if there is a consistency in sensitivity predictions. Also point out why Sobol method is considered to be somewhat of a pinnacle of SA (i.e. variance-based SA are considered to be very accurate) and say how this variance-based approach is different to the coefficients-based SA where applicable. 
           - Explanation of high-impact parameters from both mathematical and physical perspectives.
           - Be very mathematical in your analysis and think like an expert in UQ and SA - details are important.
           - Discussion on parameters with minimal impact and why in terms of the actual physics.
           - Discussion on correlation coefficients that are negative and what does it mean in terms of the actual model. 
           - Discussion on correlation coefficients that are consistently higher than other ones and why. 
           - Discussion if the Sobol index bounds are large or small and what does it mean mathematically and physically!
           - In this section you must make reference to correlation_coefficients.png and sobol_indices.png 
           - As a reminder that will help you and the user (you must communicate the principles below when you talk about Sobol indices so that the user has better context - actually genereate the equations in latex and contextualize them): There exist several types of Sobol indices. The first order Sobol sensitivity index $S$ measures the direct effect each parameter has on the variance of the model:
            
            $$
            S_i=\\frac{{\\mathbb{{V}}\\left[\\mathbb{{E}}\\left[Y \\mid Q_i\\right]\\right}}{{\\mathbb{{V}}[Y]}}
            $$

            Here, $\\mathbb{{E}}\\left[Y \\mid Q_i\\right]$ denotes the expected value of the output $Y$ when parameter $Q_i$ is fixed. The first order Sobol sensitivity index tells us the expected reduction in the variance of the model when we fix parameter $Q_i$. The sum of the first order Sobol sensitivity indices can not exceed one (Glen and Isaacs, 2012).

            Higher order sobol indices exist, and give the sensitivity due interactions between a parameter $Q_i$ and various other parameters. It is customary to only calculate the first and total order indices (Saltelli et al., 2010). The total Sobol sensitivity index $S_{{T i}}$ includes the sensitivity of both first order effects as well as the sensitivity due to interactions (covariance) between a given parameter $Q_i$ and all other parameters (Homma and Saltelli, 1996). It is defined as:
            $$
            S_{{T i}}=1-\\frac{{\\mathbb{{V}}\\left[\\mathbb{{E}}\\left[Y \\mid Q_{{-i}}\\right]\\right}}{{\\mathbb{{V}}[Y]}},
            $$
            where $Q_{{-i}}$ denotes all uncertain parameters except $Q_i$. The sum of the total Sobol sensitivity indices is equal to or greater than one (Glen and Isaacs, 2012). If no higher order interactions are present, the sum of both the first and total order sobol indices are equal to one.
            
           
        4. **Key Findings:**
           - Critical insights from the analysis.

        5. **Conclusion:**
           - Recommendations for further refinement and investigation.

        6. **Summary and Insights for Decision Making:**
           - Critical influence of key parameters.
           - Management focus on controlling influential parameters.
           - Implications of negative correlations.
           - Use of sensitivity analysis for targeted interventions.

        ### Figures
        The figures will be located in the `figures` directory with the following names (you must never omit these figures - they must always be referenced in the report):
        - `correlation_coefficients.png`: Correlation coefficients plot.
        - `grid_plot.png`: Grid plot of input parameters against model output (one-by-one with all other params fixed).
        - `mean_estimate_convergence_plot.png`: Mean estimate convergence plot.
        - `sobol_indices.png`: Sobol' indices plot.
        Also, you must make references to the figures directly from the report text in the right place with the right premise, i.e. ... as can be seen in Figure x ... 


        ### LaTeX Report Format
        Generate a LaTeX report based on the above structure and include the figures using the provided filenames. Ensure to refer to figures explicitly in the text and include tables for the correlation coefficients and Sobol indices (both first and total order) from the CSV data.

        Please provide the LaTeX code for the report. Only respond in latex, immediately with the report, no need for any typical openings etc, just raw latex pls. The quality has to be that of a report that could be published in a reputable scientific journal.
        Never refer to the actual names of the .csv files in the report.


        """

        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        app.logger.error(f"Error in call_chatgpt_analysis: {e}")
        app.logger.error(traceback.format_exc())
        raise

# def generate_pdf_from_latex(latex_code):
#     pdf_path = "report.pdf"
#     tex_path = "report.tex"
    
#     # Add basic LaTeX document structure if not present
#     if not latex_code.strip().startswith("\\documentclass"):
#         latex_code = "\\documentclass{article}\n" \
#                      "\\usepackage{amsmath}\n" \
#                      "\\usepackage{geometry}\n" \
#                      "\\usepackage{graphicx}\n" \
#                      "\\usepackage{booktabs}\n" \
#                      "\\geometry{margin=2cm}\n" \
#                      "\\begin{document}\n" \
#                      + latex_code + \
#                      "\n\\end{document}"
    
#     with open(tex_path, "w") as tex_file:
#         tex_file.write(latex_code)
    
#     # Run pdflatex to generate the PDF
#     os.system(f"pdflatex -interaction=nonstopmode {tex_path}")
#     os.system(f"pdflatex -interaction=nonstopmode {tex_path}")  # Running twice for proper cross-references
    
#     if os.path.exists("report.pdf"):
#         shutil.move("report.pdf", pdf_path)
#     else:
#         raise FileNotFoundError("report.pdf not found")
    
#     return pdf_path


import PyPDF2

def generate_pdf_from_latex(latex_code):
    pdf_path = "report.pdf"
    tex_path = "report.tex"
    temp_pdf_path = "temp_report.pdf"

    # Add basic LaTeX document structure if not present
    if not latex_code.strip().startswith("\\documentclass"):
        latex_code = "\\documentclass{article}\n" \
                     "\\usepackage{amsmath}\n" \
                     "\\usepackage{geometry}\n" \
                     "\\usepackage{graphicx}\n" \
                     "\\usepackage{booktabs}\n" \
                     "\\usepackage{hyperref}\n" \
                     "\\geometry{margin=2cm}\n" \
                     "\\pagestyle{empty}\n" \
                     "\\pagenumbering{gobble}\n" \
                     "\\begin{document}\n" \
                     + latex_code + \
                     "\n\\end{document}"
    
    with open(tex_path, "w") as tex_file:
        tex_file.write(latex_code)
    
    # Run pdflatex to generate the PDF
    os.system(f"pdflatex -interaction=nonstopmode -output-directory . {tex_path}")
    os.system(f"pdflatex -interaction=nonstopmode -output-directory . {tex_path}")  # Running twice for proper cross-references
    
    # Move the generated PDF to a temporary path
    if os.path.exists("report.pdf"):
        shutil.move("report.pdf", temp_pdf_path)
    else:
        raise FileNotFoundError("report.pdf not found")

    # Remove the first page from the PDF using PdfReader and PdfWriter
    with open(temp_pdf_path, "rb") as input_pdf:
        reader = PyPDF2.PdfReader(input_pdf)
        writer = PyPDF2.PdfWriter()
        for page_num in range(1, len(reader.pages)):  # Skip the first page
            writer.add_page(reader.pages[page_num])

        with open(pdf_path, "wb") as output_pdf:
            writer.write(output_pdf)

    # Clean up temporary files
    os.remove(tex_path)
    os.remove(temp_pdf_path)

    return pdf_path



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Disable reloader to prevent restarting on file changes
