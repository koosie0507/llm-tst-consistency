def format_value(value, is_different):
    if is_different:
        if value > 0:
            return f"\\textbf{{\\underline{{{value}}}}}"
        elif value < 0:
            return f"\\textbf{{\\textit{{{value}}}}}"
    return str(value)


def make_latex_table(df, caption, label, features, alignment):
    # Start the table and specify the alignment
    latex_table = "\\begin{table}[!ht]\n\\centering\n"
    latex_table += f"\\resizebox{{\\textwidth}}{{!}}{{%\n\\begin{{tabular}}{{{''.join(alignment)}}}\n"

    # Add the column headers
    headers = [f"{feat}_diff" for feat in features]
    latex_table += " & ".join(map(lambda x: x.replace("_", "\\_"), ["model"] + features)) + " \\\\\n\\toprule\n"

    # Process each row
    table_rows = []
    for index, row in df.iterrows():
        row_values = [index.replace("_", "\\_")]
        for col in headers:
            # Get the corresponding significance column name
            significance_col = f"is_{col}erent"
            # Format the value based on the significance
            formatted_value = format_value(row[col], row[significance_col])
            row_values.append(formatted_value)
        table_rows.append(" & ".join(row_values) + " \\\\ ")
    latex_table += "\\midrule\n".join(table_rows)

    # End the table
    latex_table += "\\bottomrule\n\\end{tabular}}\n"
    latex_table += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    latex_table += "\\end{table}"

    return latex_table
