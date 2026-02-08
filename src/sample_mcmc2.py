import os
from typing import *
from numpyro.infer import MCMC, Predictive, init_to_value, NUTS
from jax import numpy as jnp
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.paths import DATA_DIRECTORY, EXPERIMENT_DIRECTORY

matplotlib.use('TkAgg')

class SampleMCMC2:

    def __init__(self, name: str, amino_acid: str, n_parameters, n_components: int, data, model, kernel_type, rng_key, density_fn, num_warmup: int=500, num_samples: int=1000, test_data=None, fold=None):
        self.name = name
        self.amino_acid = amino_acid
        self.n_parameters = n_parameters
        self.n_components = n_components
        self.data = data
        self.model = model
        self.kernel_type = kernel_type
        self.rng_key = rng_key
        self.density_fn = density_fn
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.test_data = test_data
        self.fold = fold

    def initialize_parameters(self):
        self.kmeans = KMeans(self.n_components)
        self.kmeans.fit(self.data)
        self.init_locs = {
            "phi_loc": self.kmeans.cluster_centers_[:, 0],
            "psi_loc": self.kmeans.cluster_centers_[:, 1],
        }

    def run_mcmc(self):
        self.kernel = NUTS(
            self.model,
            init_strategy=init_to_value(values=self.init_locs),
            max_tree_depth=7
            )
        self.mcmc = MCMC(self.kernel, num_warmup=self.num_warmup, num_samples=self.num_samples)
        self.mcmc.run(self.rng_key, data=self.data, num_data=len(self.data), n_components=self.n_components)
        self.posterior_samples = self.mcmc.get_samples()

    def predict(self):
        # Posterior Predictive Check
        self.predictive = Predictive(self.model, posterior_samples=self.posterior_samples, parallel=True)
        self.predicted_samples = self.predictive(self.rng_key, None, 1, self.n_components)

    def estimate_params(self):
        self.estimated_params = {}
        for parameter in self.posterior_samples:
            self.estimated_params[parameter] = jnp.mean(self.posterior_samples[parameter], axis=0)

    @property
    def density(self):
        return self.density_fn(self.data, self.estimated_params, self.num_samples, self.n_components)

    @property
    def loglikelihood(self):
        return jnp.sum(jnp.log(self.density))

    @property
    def aic(self):
        return -2 * self.loglikelihood + 2 * self.n_parameters * self.n_components

    @property
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

    @property
    def test_loglikelihood(self):
        return jnp.sum(jnp.log(self.density_fn(self.test_data, self.estimated_params, self.num_samples, self.n_components)))

    @property
    def test_aic(self):
        return -2 * self.test_loglikelihood + 2 * self.n_parameters * self.n_components

    @property
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

        # Scatter plot of actual vs predicted phi
        plt.figure()
        plt.grid()
        sns.kdeplot(x=predicted_phi, y=predicted_psi, z=self.density, label='Predicted', alpha=0.5, color='red')
        plt.scatter(self.data[:, 0], self.data[:, 1], alpha=0.1, label='Actual', color='blue')
        plt.scatter(self.estimated_params["phi_loc"], self.estimated_params["psi_loc"], alpha=1, label='Component means', color='black', marker='x')
        plt.title(self.amino_acid)
        plt.legend()
        plt.ylim(-4, 4)
        plt.xlim(-4, 4)

        directory_path = os.path.join(EXPERIMENT_DIRECTORY, self.name, self.amino_acid, f"{self.n_components}_components")
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        if self.fold is not None:
            plt.savefig(os.path.join(directory_path, f"{self.n_components}_components_fold_{self.fold}_{self.amino_acid}_contour.png"))
        else:
            plt.savefig(os.path.join(directory_path, f"{self.n_components}_components_{self.amino_acid}_contour.png"))

    def plot_scatter(self):
        if "phi_psi" in self.predicted_samples:
            self.predicted_samples["phi_psi"] = self.predicted_samples["phi_psi"].reshape(-1, 2)
            predicted_phi = self.predicted_samples["phi_psi"][:, 0]
            predicted_psi = self.predicted_samples["phi_psi"][:, 1]
        else:
            predicted_phi = self.predicted_samples["phi"]
            predicted_psi = self.predicted_samples["psi"]

        # Scatter plot of actual vs predicted phi
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

        directory_path = os.path.join(EXPERIMENT_DIRECTORY, self.name, self.amino_acid, f"{self.n_components}_components")
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)

        if self.fold is not None:
            plt.savefig(os.path.join(directory_path, f"{self.n_components}_components_fold_{self.fold}_{self.amino_acid}_scatter.png"))
        else:
            plt.savefig(os.path.join(directory_path, f"{self.n_components}_components_{self.amino_acid}_scatter.png"))

    @staticmethod
    def run_for_n_components(n_components: List[int], data, model, kernel_type, rng_key, num_warmup, num_samples):
        result_list = []
        for n_component in n_components:
            sample_mcmc = SampleMCMC(
                n_components=n_component,
                data=data,
                model=model,
                kernel_type=kernel_type,
                rng_key=rng_key,
                num_warmup=num_warmup,
                num_samples=num_samples
            )

            sample_mcmc.initialize_parameters()
            sample_mcmc.run_mcmc()
            sample_mcmc.predict()

            result_list.append(sample_mcmc)


