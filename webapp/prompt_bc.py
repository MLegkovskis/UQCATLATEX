def get_prompt(user_model_code, combined_coefficients_csv, expectation_convergence_csv, first_order_sobol_csv, total_order_sobol_csv):
    return f"""
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

    1. **Model Overview:**
       - Brief description of the model and input parameters in LaTeX format - never reprint the model as python code, always convert the numpy-based model to the latex format.
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
    Generate a LaTeX report based on the above structure and include the figures using the provided filenames. Set margins to 1.5 cm. Ensure to refer to figures explicitly in the text and include tables for the correlation coefficients and Sobol indices (both first and total order) from the CSV data.

    Please provide the LaTeX code for the report. Only respond in latex, immediately with the report, no need for any typical openings etc, just raw latex pls. The quality has to be that of a report that could be published in a reputable scientific journal.
    Never refer to the actual names of the .csv files in the report.
    """
