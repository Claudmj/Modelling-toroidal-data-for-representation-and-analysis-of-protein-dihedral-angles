from jax import random
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime

from src.paths import EXPERIMENT_DIRECTORY, DATA_DIRECTORY
from src.bivariate_von_mises_models import BivariateVonMisesModels
from src.bivariate_von_mises_utils import BivariateVonMisesUtils
from src.sample_mcmc2 import SampleMCMC2
from src.nine_mers_dataset2 import NineMersDataset2
from src.utils import Utils


nine_mers = NineMersDataset2()

for amino_acid in nine_mers.AMINO_ACIDS:
    average_result_dict = Utils.make_metric_result_dict()
    residue_angles_array = np.array(nine_mers.dataset[amino_acid])
    amino_acid_directory = os.path.join(EXPERIMENT_DIRECTORY, "product_kfold2", amino_acid)
    if not os.path.isdir(amino_acid_directory):
        os.mkdir(amino_acid_directory)

    with open(os.path.join(DATA_DIRECTORY, "splits", f"{amino_acid}_5_split.json"), "rt") as file:
        shuffle = json.load(file)

    for i in range(1, 11):
        component_result_dict = Utils.make_fold_result_dict()

        for fold, (train, test) in enumerate(shuffle):
            sample_mcmc = SampleMCMC2(
                name="product_kfold",
                amino_acid=amino_acid,
                n_parameters=5,
                n_components=i,
                data=residue_angles_array[train],
                model=BivariateVonMisesModels.product_von_mises,
                kernel_type="nuts",
                rng_key=random.PRNGKey(123),
                density_fn=BivariateVonMisesUtils.product_density,
                num_warmup=500,
                num_samples=1000,
                test_data=residue_angles_array[test],
                fold=fold + 1
            )

            start_time = datetime.now()
            sample_mcmc.initialize_parameters()
            sample_mcmc.run_mcmc()
            sample_mcmc.predict()
            sample_mcmc.estimate_params()
            end_time = datetime.now()
            time_difference = end_time - start_time
            time_difference = time_difference.total_seconds()
            sample_mcmc.plot_scatter()
            sample_mcmc.plot_contour()
            print(sample_mcmc.loglikelihood, sample_mcmc.aic, sample_mcmc.bic)
            Utils.add_to_fold_result_dict(component_result_dict, sample_mcmc, time_difference)

        output_folder_name = os.path.join(amino_acid_directory, f"{i}_components")
        output_file_name = os.path.join(output_folder_name, f"{i}_components_kfold.csv")

        if not os.path.isdir(output_folder_name):
            os.mkdir(output_folder_name)
        means = Utils.summary_stats_result_dict(f"{amino_acid} {i} components", component_result_dict, output_file_name)
        Utils.add_to_metric_result_dict(average_result_dict, means)
    df = pd.DataFrame(average_result_dict)
    df.to_csv(os.path.join(amino_acid_directory, f"{amino_acid}_kfold.csv"))
print(0)
