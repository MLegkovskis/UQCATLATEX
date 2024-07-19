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
       - Try to deduce units of all the variables including the model output and use them for better clarity
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

       - Use the following table for the correlation coefficients (you must copy-paste this table into the report as is and use to contextualise the data):

        \\begin{{table}}[h]
        \\centering
        \\begin{{tabular}}{{|p{{2cm}}|c|p{{12cm}}|}}
        \\hline
        \\textbf{{Coefficient}} & \\textbf{{Mathematics}} & \\textbf{{Description}} \\\\
        \\hline
        PCC (Pearson) & $\\rho_{{U,V}} = \\frac{{\\text{{Cov}}(U,V)}}{{\\sigma_U \\sigma_V}}$ & Measures the strength of a linear relationship between two variables. Can be positive or negative. Values range from -1 to 1, where 1 indicates a perfect positive linear relationship and -1 indicates a perfect negative linear relationship. Zero does not necessarily imply independence. If PCC for a given model input is 0.5, it means there is a moderate positive linear relationship with the output. \\\\
        \\hline
        PRCC (Partial Rank Correlation Coefficient) & $\\widehat{{\\rho}}^S_{{U,V}}$ (on ranked data) & Computes the Pearson correlation coefficient on ranked input and output variables. Useful for identifying monotonic relationships when linearity is not present. Values range from -1 to 1. If PRCC for a given model input is 0.5, it means there is a moderate positive monotonic relationship with the output. \\\\
        \\hline
        SRC (Standard Regression Coefficient) & $\\widehat{{\\text{{SRC}}}}_i = \\widehat{{a}}_i \\frac{{\\widehat{{\\sigma}}_i}}{{\\widehat{{\\sigma}}}}$ & Measures the influence of input variables on output using multiple linear regression. Useful for linear relationships. The closer to 1, the greater the impact on the variance of $Y$. Negative values are possible. If SRC for a given model input is 0.5, it means the input has a moderate positive influence on the output variance. \\\\
        \\hline
        SRRC (Standard Rank Regression Coefficient) & $\\widehat{{\\text{{SRC}}}}_i$ (on ranked data) & Computes SRC on ranked input and output variables. Useful for monotonic relationships where linearity is not present. Values range from -1 to 1. If SRRC for a given model input is 0.5, it means the input has a moderate positive monotonic influence on the output. \\\\
        \\hline
        Spearman & $\\rho^S_{{U,V}} = \\rho_{{F_U(U),F_V(V)}}$ & Measures the strength of a monotonic relationship between two variables using ranks. Equivalent to Pearson's on ranked data. Values range from -1 to 1, indicating perfect monotonic relationships. Can be positive or negative. If Spearman for a given model input is 0.5, it means there is a moderate positive monotonic relationship with the output. \\\\
        \\hline
        \\multicolumn{{3}}{{|p{{17cm}}|}}{{\\textbf{{Variables:}} $\\text{{Cov}}(U,V)$: covariance between $U$ and $V$; $\\sigma_U, \\sigma_V$: standard deviations of $U$ and $V$; $\\widehat{{a}}_i$: estimated regression coefficients; $\\widehat{{\\sigma}}_i, \\widehat{{\\sigma}}$: sample standard deviations; $F_U, F_V$: cumulative distribution functions; $U, V$: random variables; $\\rho$: correlation coefficient.}} \\\\
        \\hline
        \\end{{tabular}}
        \\caption{{Summary of Sensitivity Analysis Coefficients}}
        \\end{{table}}

       - Use the following text to explain Sobol indices (you must copy-paste this table into the report as is and use to contextualise the data):

        Sobol' indices are a powerful method in uncertainty quantification (UQ) and sensitivity analysis (SA) that measure the influence of input variables on the output of a model. Consider an input random vector $\\mathbf{{X}} = \\left( X_1, \\ldots, X_d \\right)$ and an output $Y$ of the model. The Sobol' method decomposes the variance of $Y$ into contributions from each input variable and their interactions.

        The Sobol' decomposition is given by:
        \\[ Y = h_0 + \\sum_{{i=1}}^d h_{{\\{{i\\}}}}(X_i) + \\sum_{{1 \\leq i < j \\leq d}} h_{{\\{{i,j\\}}}}(X_i, X_j) + \\ldots + h_{{\\{{1, 2, \\ldots, d\\}}}}(X_1, X_2, \\ldots, X_d), \\]
        where $h_0$ is a constant and $h_{{\\{{i\\}}}}$, $h_{{\\{{i,j\\}}}}$, etc. are functions capturing the effect of individual variables and their interactions.

        The first order Sobol' index $S_i$ measures the contribution of a single input variable $X_i$ to the variance of $Y$:
        \\[ S_i = \\frac{{\\Var{{\\Expect{{Y | X_i}}}}}}{{\\Var{{Y}}}}, \\]
        indicating how much of the output variance can be explained by $X_i$ alone. A first order Sobol' index of 0.5 means that 50\\% of the output variance is due to $X_i$.

        The total Sobol' index $S^T_i$ accounts for both the individual effect of $X_i$ and its interactions with other variables:
        \\[ S^T_i = 1 - \\frac{{\\Var{{\\Expect{{Y | X_{{\\overline{{\\{{i\\}}}}}}}}}}}}{{\\Var{{Y}}}}, \\]
        where $X_{{\\overline{{\\{{i\\}}}}}}$ denotes all input variables except $X_i$. A total Sobol' index of 0.5 means that 50\\% of the output variance involves $X_i$ through its own effect and its interactions.

        Unlike correlation coefficients (PCC, PRCC, Spearman, SRC, SRRC), which measure linear or monotonic relationships between inputs and outputs, Sobol' indices provide a comprehensive variance-based decomposition, capturing both individual and interaction effects. While correlation coefficients can identify linear or monotonic influence, they do not quantify the contribution to output variance or the interaction effects. Therefore, Sobol' indices and correlation coefficients complement each other in UQ and SA: correlation coefficients help identify relationships, whereas Sobol' indices quantify their contributions to output variability.

        In summary, Sobol' indices offer a detailed variance-based sensitivity analysis, distinguishing between first order effects and total effects, which include interactions. They are essential for understanding the contributions of input variables to model output variance and provide insights beyond what correlation coefficients can offer.

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