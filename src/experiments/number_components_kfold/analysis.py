import os
import pandas as pd
import matplotlib.pyplot as plt
from math import log

from src.paths import ANALYSIS_DIRECTORY, EXPERIMENT_DIRECTORY
from src.plots import Plots
from src.nine_mers_dataset2 import NineMersDataset2


nine_mers = NineMersDataset2()

for residue in nine_mers.AMINO_ACIDS:
# residue = "V"
    distributions = ['Product von Mises', 'Sine bivariate von Mises', 'Sine skewed sine bivariate von Mises']
    product_result_path = os.path.join(EXPERIMENT_DIRECTORY, "product_5x5fold", residue, f"{residue}_grand_summary_25folds.csv")
    sine_result_path = os.path.join(EXPERIMENT_DIRECTORY, "sine_5x5fold", residue, f"{residue}_grand_summary_25folds.csv")
    skewed_sine_result_path = os.path.join(EXPERIMENT_DIRECTORY, "sine_skewed_5x5fold", residue, f"{residue}_grand_summary_25folds.csv")

    times = []
    loglikelihoods = []
    aic = []
    bic = []
    residue_analysis_directory = os.path.join(ANALYSIS_DIRECTORY)
    for i, path in enumerate([product_result_path, sine_result_path, skewed_sine_result_path]):
        df = pd.read_csv(path)
        times.append(df['mean_time'].tolist()[1:])
        aic.append(df['mean_aic'].tolist()[1:])
        bic.append(df['mean_bic'].tolist()[1:])
        if residue == "V":
            Plots.elbow_plot_single(df['mean_time'].tolist()[1:], df['mean_aic'].tolist()[1:], df['mean_bic'].tolist()[1:], os.path.join(residue_analysis_directory, f"elbow_plot_{distributions[i]}.png"))

    if not os.path.isdir(residue_analysis_directory):
        os.mkdir(residue_analysis_directory)


    # Plots.elbow_plot(times, aic, bic, distributions, os.path.join(residue_analysis_directory, f"{residue}_elbow_plot.png"))
    # print(0)