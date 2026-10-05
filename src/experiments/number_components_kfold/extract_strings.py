import pandas as pd 
selected_components = 9
# csv_path = "data/experiments/product_5x5fold/V/V_grand_summary_25folds.csv"
# csv_path = "data/experiments/sine_5x5fold/V/V_grand_summary_25folds.csv"
csv_path = "data/experiments/sine_skewed_5x5fold/V/V_grand_summary_25folds.csv"


def extract_row_for_components(csv_path, selected_components):
    df = pd.read_csv(csv_path)
    selected_row = df[df.iloc[:, 0] == selected_components]

    if selected_row.empty:
        raise ValueError(f"Could not find a row for {selected_components} components in {csv_path}")

    header_line = " & ".join([
        "Number of components",
        "log-likelihood",
        "log-likelihood SD",
        "AIC",
        "AIC SD",
        "BIC",
        "BIC SD",
        "Time",
        "SD (s)",
    ])

    row_line = str(selected_components)
    for col in ['loglik', 'aic', 'bic', 'time']:
        mean = selected_row[f"mean_{col}"].iloc[0]
        std = selected_row[f"sd_{col}"].iloc[0]
        row_line += f" & {mean:.2f} $\\pm$ {std:.2f}"

    return header_line, row_line


if __name__ == "__main__":
    header_line, row_line = extract_row_for_components(csv_path, selected_components)
    print(header_line)
    print(row_line)