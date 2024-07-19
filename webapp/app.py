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
import PyPDF2
from analysis_script import run_analysis_script


app = Flask(__name__)
OUTPUT_API_DIR = "results_api"
EXAMPLES_DIR = "examples"


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

# def generate_pdf_from_latex(latex_code):
#     pdf_path = "report.pdf"
#     tex_path = "report.tex"
#     temp_pdf_path = "temp_report.pdf"

#     # Add basic LaTeX document structure if not present
#     if not latex_code.strip().startswith("\\documentclass"):
#         latex_code = "\\documentclass{article}\n" \
#                      "\\usepackage{amsmath}\n" \
#                      "\\usepackage{geometry}\n" \
#                      "\\usepackage{graphicx}\n" \
#                      "\\usepackage{booktabs}\n" \
#                      "\\usepackage{hyperref}\n" \
#                      "\\geometry{margin=2cm}\n" \
#                      "\\pagestyle{empty}\n" \
#                      "\\pagenumbering{gobble}\n" \
#                      "\\begin{document}\n" \
#                      + latex_code + \
#                      "\n\\end{document}"
    
#     with open(tex_path, "w") as tex_file:
#         tex_file.write(latex_code)
    
#     # Run pdflatex to generate the PDF
#     os.system(f"pdflatex -interaction=nonstopmode -output-directory . {tex_path}")
#     os.system(f"pdflatex -interaction=nonstopmode -output-directory . {tex_path}")  # Running twice for proper cross-references
    
#     # Move the generated PDF to a temporary path
#     if os.path.exists("report.pdf"):
#         shutil.move("report.pdf", temp_pdf_path)
#     else:
#         raise FileNotFoundError("report.pdf not found")

#     # Remove the first page from the PDF using PdfReader and PdfWriter
#     with open(temp_pdf_path, "rb") as input_pdf:
#         reader = PyPDF2.PdfReader(input_pdf)
#         writer = PyPDF2.PdfWriter()
#         for page_num in range(1, len(reader.pages)):  # Skip the first page
#             writer.add_page(reader.pages[page_num])

#         with open(pdf_path, "wb") as output_pdf:
#             writer.write(output_pdf)

#     # Clean up temporary files
#     os.remove(tex_path)
#     os.remove(temp_pdf_path)

#     return pdf_path


def generate_pdf_from_latex(latex_code):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_path = os.path.join(output_dir, "report.pdf")
    tex_path = os.path.join(output_dir, "report.tex")
    temp_pdf_path = os.path.join(output_dir, "temp_report.pdf")

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
    os.system(f"pdflatex -interaction=nonstopmode -output-directory {output_dir} {tex_path}")
    os.system(f"pdflatex -interaction=nonstopmode -output-directory {output_dir} {tex_path}")  # Running twice for proper cross-references
    
    # Move the generated PDF to a temporary path
    if os.path.exists(pdf_path):
        shutil.move(pdf_path, temp_pdf_path)
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
    os.remove(temp_pdf_path)

    return pdf_path




if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Disable reloader to prevent restarting on file changes
