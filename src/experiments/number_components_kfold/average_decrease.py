# from pathlib import Path
# import pandas as pd

# base_dir = Path("data/experiments")

# models = {
#     "product": "product_5x5fold",
#     "sine": "sine_5x5fold",
#     "skewed": "sine_skewed_5x5fold",
# }

# residues = [
#     "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
#     "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
# ]

# reference_model = "product"
# comparators = ["skewed"]

# all_aic_pct_drops = []
# all_time_pct_drops = []

# for residue in residues:
#     ref_path = (
#         base_dir
#         / models[reference_model]
#         / residue
#         / f"{residue}_grand_summary_25folds.csv"
#     )
#     ref_df = pd.read_csv(ref_path)

#     for comp_model in comparators:
#         comp_path = (
#             base_dir
#             / models[comp_model]
#             / residue
#             / f"{residue}_grand_summary_25folds.csv"
#         )
#         comp_df = pd.read_csv(comp_path)

#         merged = ref_df.merge(
#             comp_df, on="n_components", suffixes=("_ref", f"_{comp_model}")
#         )

#         comp_aic = merged[f"mean_aic_{comp_model}"]
#         comp_time = merged[f"mean_time_{comp_model}"]

#         # Positive value indicates a reduction relative to the comparator baseline
#         # Formula: (comparator - reference) / comparator * 100
#         aic_pct_drop = (
#             (comp_aic - merged["mean_aic_ref"]) / comp_aic.abs() * 100
#         ).mean()
#         time_pct_drop = (
#             (comp_time - merged["mean_time_ref"]) / comp_time * 100
#         ).mean()

#         all_aic_pct_drops.append(aic_pct_drop)
#         all_time_pct_drops.append(time_pct_drop)

# avg_aic_pct_decrease = sum(all_aic_pct_drops) / len(all_aic_pct_drops)
# avg_time_pct_decrease = sum(all_time_pct_drops) / len(all_time_pct_drops)

# print(
#     f"Average AIC % decrease vs other models: {avg_aic_pct_decrease:.2f}%"
# )
# print(
#     f"Average time % decrease vs other models: {avg_time_pct_decrease:.2f}%"
# )


import os
import shutil

optimal_k = {
    'A': (4, 'sbvm'),
    'R': (3, 'ssbvm'),
    'N': (3, 'ssbvm'),
    'D': (3, 'ssbvm'),
    'C': (3, 'ssbvm'),
    'Q': (4, 'ssbvm'),
    'E': (5, 'sbvm'),
    'G': (4, 'ssbvm'),
    'H': (3, 'ssbvm'),
    'I': (8, 'sbvm'),
    'L': (3, 'ssbvm'),
    'K': (4, 'ssbvm'),
    'M': (4, 'ssbvm'),
    'F': (4, 'ssbvm'),
    'P': (6, 'sbvm'),
    'S': (6, 'pvm'),
    'T': (5, 'sbvm'),
    'W': (5, 'sbvm'),
    'Y': (4, 'ssbvm'),
    'V': (9, 'ssbvm')
}

for RESIDUE, (k_opt, model) in optimal_k.items():
    full_model = {
        'pvm': 'product',
        'sbvm': 'sine',
        'ssbvm': 'sine_skewed_sine'
    }[model]
    other_model = {
        'pvm': 'sine',
        'sbvm': 'sine_skewed',
        'ssbvm': 'product'
    }[model]

    contour_plot_path = os.path.join("data", "experiments", f"{other_model}_5x5fold", RESIDUE, "split_0", f"{k_opt}_components", f"{k_opt}_components_fold_1_{RESIDUE}_contour.png")
    new_contour_plot_path = os.path.join("data", "analysis", "contour_plots", f"{full_model}_{RESIDUE}_contour_plot_k{k_opt}.png")
    shutil.copy(contour_plot_path, new_contour_plot_path)

    scatter_plot_path = os.path.join("data", "experiments", f"{other_model}_5x5fold", RESIDUE, "split_0", f"{k_opt}_components", f"{k_opt}_components_fold_1_{RESIDUE}_scatter.png")
    new_scatter_plot_path = os.path.join("data", "analysis", "scatter_plots", f"{full_model}_{RESIDUE}_scatter_plot_k{k_opt}.png")
    shutil.copy(scatter_plot_path, new_scatter_plot_path)
