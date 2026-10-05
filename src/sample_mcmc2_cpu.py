import os
import gc
from functools import cached_property
from typing import *

# 1. Optimize CPU environment variables BEFORE importing JAX/NumPyro
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1" # Prevent OMP from fighting with NumPyro parallel chains

import numpyro
CPU_CORES = 4 # Set this to match your physical CPU cores
numpyro.set_platform("cpu")
numpyro.set_host_device_count(CPU_CORES)

import jax
from jax import numpy as jnp
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)

from numpyro.infer import MCMC, Predictive, init_to_value, NUTS
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.paths import DATA_DIRECTORY, EXPERIMENT_DIRECTORY

matplotlib.use('TkAgg')

class SampleMCMC2CPU:

    def __init__(self, name: str, amino_acid: str, n_parameters, n_components: int, data, model, kernel_type, rng_key, density_fn, num_warmup: int=500, num_samples: int=1000, test_data=None, fold=None, split=None):
        self.name = name
        self.amino_acid = amino_acid
        self.n_parameters = n_parameters
        self.n_components = n_components
        # Cast to JAX array once to avoid repeated conversions
        self.data = jnp.array(data)
        self.model = model
        self.kernel_type = kernel_type
        self.rng_key = rng_key
        self.density_fn = density_fn
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.test_data = jnp.array(test_data) if test_data is not None else None
        self.fold = fold
        self.split = split
        self.directory_path = os.path.join(EXPERIMENT_DIRECTORY, self.name, self.amino_acid, self.split, f"{self.n_components}_components")
        os.makedirs(self.directory_path, exist_ok=True)


    def initialize_parameters(self):
        self.kmeans = KMeans(self.n_components, n_init='auto')
        # sklearn requires standard numpy arrays
        self.kmeans.fit(np.array(self.data))
        self.init_locs = {
            "phi_loc": jnp.array(self.kmeans.cluster_centers_[:, 0]),
            "psi_loc": jnp.array(self.kmeans.cluster_centers_[:, 1]),
        }

    def run_mcmc(self):
        self.kernel = NUTS(
            self.model,
            init_strategy=init_to_value(values=self.init_locs),
            max_tree_depth=6
        )
        # Running 4 chains in parallel on 4 CPU cores (1/4th the samples per chain)
        self.mcmc = MCMC(
            self.kernel, 
            num_warmup=self.num_warmup, 
            num_samples=self.num_samples, 
            num_chains=1, 
            chain_method="sequential", 
            progress_bar=False
        )  
        self.mcmc.run(self.rng_key, data=self.data, num_data=len(self.data), n_components=self.n_components)

    def predict(self):
        # Posterior Predictive Check
        self.predictive = Predictive(self.model, posterior_samples=self.mcmc.get_samples(), return_sites=['phi_psi', 'phi', 'psi'], parallel=False)
        pred = self.predictive(self.rng_key, None, 1, self.n_components)

        self.predicted_samples = jax.device_get(pred)
        del pred
        gc.collect()
        # jax.clear_caches() removed for compilation speed

    def estimate_params(self):
        posterior = self.mcmc.get_samples()
        self.estimated_params = {k: jnp.mean(v, axis=0) for k, v in posterior.items()}

        del posterior
        gc.collect()
        # jax.clear_caches() removed for compilation speed

    @cached_property
    def density(self):
        return self.density_fn(self.data, self.estimated_params, self.num_samples, self.n_components)

    @cached_property
    def loglikelihood(self):
        return jnp.sum(jnp.log(self.density))

    @cached_property
    def aic(self):
        return -2 * self.loglikelihood + 2 * self.n_parameters * self.n_components

    @cached_property
    def bic(self):
        return -2 * self.loglikelihood + self.n_parameters * self.n_components * jnp.log(self.num_samples)

    def calculate_loglikelihood(self, data):
        return jnp.sum(jnp.log(self.density_fn(data, self.estimated_params, self.num_samples, self.n_components)))

    def calculate_aic(self, data):
        loglikelihood = self.calculate_loglikelihood(data)
        return -2 * loglikelihood + 2 * self.n_parameters * self.n_components

    def calculate_bic(self, data):
        loglikelihood = self.calculate_loglikelihood(data)
        return -2 * loglikelihood + self.n_parameters * self.n_components * jnp.log(self.num_samples)

    @cached_property
    def test_loglikelihood(self):
        return jnp.sum(jnp.log(self.density_fn(self.test_data, self.estimated_params, self.num_samples, self.n_components)))

    @cached_property
    def test_aic(self):
        return -2 * self.test_loglikelihood + 2 * self.n_parameters * self.n_components

    @cached_property
    def test_bic(self):
        return -2 * self.test_loglikelihood + self.n_parameters * self.n_components * jnp.log(self.num_samples)

    def plot_contour(self):
        if "phi_psi" in self.predicted_samples:
            self.predicted_samples["phi_psi"] = self.predicted_samples["phi_psi"].reshape(-1, 2)
            predicted_phi = self.predicted_samples["phi_psi"][:, 0]
            predicted_psi = self.predicted_samples["phi_psi"][:, 1]
        else:
            predicted_phi = self.predicted_samples["phi"].reshape(-1)
            predicted_psi = self.predicted_samples["psi"].reshape(-1)

        plt.figure()
        plt.grid()
        # plt.hist2d(predicted_phi, predicted_psi, bins=50, cmap='Reds', alpha=0.5, density=True)
        sns.kdeplot(x=predicted_phi, y=predicted_psi, z=self.density, label='Predicted', alpha=0.5, color='red')
        plt.scatter(self.data[:, 0], self.data[:, 1], alpha=0.1, label='Actual', color='blue')
        plt.scatter(self.estimated_params["phi_loc"], self.estimated_params["psi_loc"], alpha=1, label='Component means', color='black', marker='x')
        plt.title(self.amino_acid)
        plt.legend()
        plt.ylim(-4, 4)
        plt.xlim(-4, 4)

        filename = f"{self.n_components}_components_fold_{self.fold}_{self.amino_acid}_contour.png" if self.fold is not None else f"{self.n_components}_components_{self.amino_acid}_contour.png"
        plt.savefig(os.path.join(self.directory_path, filename))
        plt.close()

    def plot_scatter(self):
        if "phi_psi" in self.predicted_samples:
            self.predicted_samples["phi_psi"] = self.predicted_samples["phi_psi"].reshape(-1, 2)
            predicted_phi = self.predicted_samples["phi_psi"][:, 0]
            predicted_psi = self.predicted_samples["phi_psi"][:, 1]
        else:
            predicted_phi = self.predicted_samples["phi"]
            predicted_psi = self.predicted_samples["psi"]

        plt.figure()
        plt.grid()
        plt.scatter(self.data[:, 0], self.data[:, 1], alpha=0.3, label='Actual', color='blue')
        plt.scatter(predicted_phi, predicted_psi, alpha=0.3, label='Predicted', color='red')
        plt.xlabel('Phi')
        plt.ylabel('Psi')
        plt.legend()
        plt.title(self.amino_acid)
        plt.ylim(-4, 4)
        plt.xlim(-4, 4)

        filename = f"{self.n_components}_components_fold_{self.fold}_{self.amino_acid}_scatter.png" if self.fold is not None else f"{self.n_components}_components_{self.amino_acid}_scatter.png"
        plt.savefig(os.path.join(self.directory_path, filename))
        plt.close()

    @staticmethod
    def run_for_n_components(n_components: List[int], data, model, kernel_type, rng_key, num_warmup, num_samples):
        result_list = []
        for n_component in n_components:
            sample_mcmc = SampleMCMC2(
                name="batch_run",
                amino_acid="Unknown",
                n_parameters=5, 
                n_components=n_component,
                data=data,
                model=model,
                kernel_type=kernel_type,
                rng_key=rng_key,
                density_fn=None, # Update if needed
                num_warmup=num_warmup,
                num_samples=num_samples
            )

            sample_mcmc.initialize_parameters()
            sample_mcmc.run_mcmc()
            sample_mcmc.predict()

            result_list.append(sample_mcmc)