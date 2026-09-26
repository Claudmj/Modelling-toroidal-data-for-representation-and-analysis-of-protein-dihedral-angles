import numpyro.distributions as dist
from jax import numpy as jnp


class BivariateVonMisesUtils:
    @staticmethod
    def product_density(data, estimated_params, num_samples, n_components):
        density = jnp.zeros([num_samples])
        for component in range(n_components):
            weight = estimated_params["mix_weights"][component]
            phi_component_distribution = dist.VonMises(loc=estimated_params["phi_loc"][component], concentration=estimated_params["phi_conc"][component])
            psi_component_distribution = dist.VonMises(loc=estimated_params["psi_loc"][component], concentration=estimated_params["psi_conc"][component])
            density += weight * jnp.exp(phi_component_distribution.log_prob(data[:, 0])) * jnp.exp(psi_component_distribution.log_prob(data[:, 1]))

        return density

    @staticmethod
    def bivariate_sine_density(data, estimated_params, num_samples, n_components):
        density = jnp.zeros([num_samples])
        for component in range(n_components):
            weight = estimated_params["mix_weights"][component]
            component_distribution = dist.SineBivariateVonMises(phi_loc=estimated_params["phi_loc"][component],
                                                           psi_loc=estimated_params["psi_loc"][component],
                                                           phi_concentration=estimated_params["phi_conc"][component],
                                                           psi_concentration=estimated_params["psi_conc"][component],
                                                           weighted_correlation=estimated_params["corr_scale"][
                                                               component])
            log_prob = component_distribution.log_prob(data)
            arr_clean = jnp.nan_to_num(log_prob, nan=-jnp.inf)
            density += weight * jnp.exp(arr_clean)

        return density

    @staticmethod
    def skewed_bivariate_sine_density(data, estimated_params, num_samples, n_components):
        density = jnp.zeros([num_samples])
        for component in range(n_components):
            weight = estimated_params["mix_weights"][component]
            sine = dist.SineBivariateVonMises(phi_loc=estimated_params["phi_loc"][component],
                                                           psi_loc=estimated_params["psi_loc"][component],
                                                           phi_concentration=estimated_params["phi_conc"][component],
                                                           psi_concentration=estimated_params["psi_conc"][component],
                                                           weighted_correlation=estimated_params["corr_scale"][component])

            skewness = estimated_params["skewness"][component]

            # Fix floating point drift by normalizing if the sum of absolute values > 1
            l1_norm = jnp.sum(jnp.abs(skewness))
            skewness_safe = jnp.where(l1_norm > 1.0, skewness / l1_norm, skewness)
            component_distribution = dist.SineSkewed(sine, skewness_safe)
            log_prob = component_distribution.log_prob(data)
            arr_clean = jnp.nan_to_num(log_prob, nan=-jnp.inf)
            density += weight * jnp.exp(arr_clean)

        return density


    @staticmethod
    def bivariate_cosine_density(data, estimated_params, num_samples, n_components):
        density = jnp.zeros([num_samples])
        for component in range(n_components):
            weight = estimated_params["mix_weights"][component]
            component_distribution = CosineBivariateVonMisesNP(phi_loc=estimated_params["phi_loc"][component],
                                                           psi_loc=estimated_params["psi_loc"][component],
                                                           phi_concentration=estimated_params["phi_conc"][component],
                                                           psi_concentration=estimated_params["psi_conc"][component],
                                                           correlation=estimated_params["corr_scale"][component])

            density += weight * jnp.exp(component_distribution.log_prob(data))

        return density