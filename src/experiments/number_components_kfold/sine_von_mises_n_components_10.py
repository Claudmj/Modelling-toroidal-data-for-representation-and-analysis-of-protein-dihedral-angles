import gc
import jax
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from jax import random

from src.paths import EXPERIMENT_DIRECTORY, DATA_DIRECTORY
from src.bivariate_von_mises_models import BivariateVonMisesModels
from src.bivariate_von_mises_utils import BivariateVonMisesUtils
from src.sample_mcmc2 import SampleMCMC2
from src.nine_mers_dataset2 import NineMersDataset2
from src.utils import Utils

nine_mers = NineMersDataset2()
NUM_SPLITS = 5  # 5 independent 5-fold cross-validations
NUM_COMPONENTS = 10

# Master base seed
master_key = random.PRNGKey(42)
split_keys = random.split(master_key, NUM_SPLITS)

for amino_acid in nine_mers.AMINO_ACIDS:
    residue_angles_array = np.array(nine_mers.dataset[amino_acid])
    
    amino_acid_directory = os.path.join(EXPERIMENT_DIRECTORY, "sine_5x5fold", amino_acid)
    os.makedirs(amino_acid_directory, exist_ok=True)
    
    all_folds_records = []

    for split_idx, s_key in enumerate(split_keys):
        split_folder_name = f"split_{split_idx}"  
        split_path = os.path.join(DATA_DIRECTORY, split_folder_name, f"{amino_acid}_5_split.json")

        if not os.path.exists(split_path):
            print(f"Warning: Split file not found at {split_path}. Skipping.")
            continue

        with open(split_path, "rt") as file:
            folds_data = json.load(file)

        split_output_dir = os.path.join(amino_acid_directory, split_folder_name)
        os.makedirs(split_output_dir, exist_ok=True)
        split_summary_dict = Utils.make_metric_result_dict()

        # Split current split_key into 5 unique deterministic keys for the 5 folds
        fold_keys = random.split(s_key, len(folds_data))

        for k in range(1, NUM_COMPONENTS + 1):  # components 1 to NUM_COMPONENTS
            component_result_dict = Utils.make_fold_result_dict()
            component_output_dir = os.path.join(split_output_dir, f"{k}_components")
            os.makedirs(component_output_dir, exist_ok=True)

            for fold, (train, test) in enumerate(folds_data):
                # Unique PRNGKey per fold derived deterministically from master_key
                fold_key = fold_keys[fold]

                sample_mcmc = SampleMCMC2(
                    name=f"sine_5x5fold",
                    amino_acid=amino_acid,
                    n_parameters=5,
                    n_components=k,
                    data=residue_angles_array[train],
                    model=BivariateVonMisesModels.sine_bivariate_von_mises,
                    kernel_type="nuts",
                    rng_key=fold_key,
                    density_fn=BivariateVonMisesUtils.bivariate_sine_density,
                    num_warmup=500,
                    num_samples=1000,
                    test_data=residue_angles_array[test],
                    fold=fold + 1,
                    split=f"split_{split_idx}"
                )

                start_time = datetime.now()
                sample_mcmc.initialize_parameters()
                sample_mcmc.run_mcmc()
                sample_mcmc.predict()
                sample_mcmc.estimate_params()
                elapsed_time = (datetime.now() - start_time).total_seconds()

                # Retained plots
                sample_mcmc.plot_scatter()
                sample_mcmc.plot_contour()

                print(sample_mcmc.loglikelihood, sample_mcmc.aic, sample_mcmc.bic)
                Utils.add_to_fold_result_dict(component_result_dict, sample_mcmc, elapsed_time)

                # Record individual fold for paired significance testing
                all_folds_records.append({
                    "amino_acid": amino_acid,
                    "split_idx": split_idx,
                    "fold_idx": fold + 1,
                    "n_components": k,
                    "loglikelihood": float(sample_mcmc.loglikelihood),
                    "aic": float(sample_mcmc.aic),
                    "bic": float(sample_mcmc.bic),
                    "time_seconds": float(elapsed_time)
                })
                # Cleanup
                del sample_mcmc
                gc.collect()
                jax.clear_caches()

            fold_csv = os.path.join(component_output_dir, f"{k}_components_fold_metrics.csv")
            means = Utils.summary_stats_result_dict(
                f"{amino_acid} {k} components split {split_idx}", 
                component_result_dict, 
                fold_csv
            )
            Utils.add_to_metric_result_dict(split_summary_dict, means)

        split_summary_df = pd.DataFrame(split_summary_dict)
        split_summary_df.to_csv(os.path.join(split_output_dir, f"{amino_acid}_split_{split_idx}_summary.csv"))

    # Consolidated 50-fold records for paired statistical tests
    full_df = pd.DataFrame(all_folds_records)
    full_df.to_csv(os.path.join(amino_acid_directory, f"{amino_acid}_all_5x5fold_metrics.csv"), index=False)

    # Aggregated grand summary across all 50 folds
    grand_summary = full_df.groupby("n_components").agg(
        mean_loglik=("loglikelihood", "mean"),
        sd_loglik=("loglikelihood", "std"),
        mean_aic=("aic", "mean"),
        sd_aic=("aic", "std"),
        mean_bic=("bic", "mean"),
        sd_bic=("bic", "std"),
        mean_time=("time_seconds", "mean"),
        sd_time=("time_seconds", "std")
    ).reset_index()

    grand_summary.to_csv(os.path.join(amino_acid_directory, f"{amino_acid}_grand_summary_25folds.csv"), index=False)
    print(f"Finished 5x5-fold CV for {amino_acid} (SBVM)")