from math import pi
import numpyro
import numpyro.distributions as dist
from jax import numpy as jnp
from numpyro.distributions.transforms import L1BallTransform
from numpyro.infer.reparam import CircularReparam


class BivariateVonMisesModels:
    @staticmethod
    @numpyro.handlers.reparam(
        config={"phi_loc": CircularReparam(), "psi_loc": CircularReparam()}
    )
    def sine_bivariate_von_mises(data, num_data, n_components):
        # Mixture prior
        mix_weights = numpyro.sample("mix_weights", dist.Dirichlet(jnp.ones((n_components,))))

        # Hprior BvM
        # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
        beta_mean_phi = numpyro.sample("beta_mean_phi", dist.Uniform(0.0, 1.0))
        beta_count_phi = numpyro.sample(
            "beta_count_phi", dist.Gamma(1.0, 1.0 / n_components)
        )  # shape, rate
        halpha_phi = beta_mean_phi * beta_count_phi
        beta_mean_psi = numpyro.sample("beta_mean_psi", dist.Uniform(0, 1.0))
        beta_count_psi = numpyro.sample(
            "beta_count_psi", dist.Gamma(1.0, 1.0 / n_components)
        )  # shape, rate
        halpha_psi = beta_mean_psi * beta_count_psi

        with numpyro.plate("mixture", n_components):
            # BvM priors

            # Place gap in forbidden region of the Ramachandran plot (protein backbone dihedral angle pairs)
            phi_loc = numpyro.sample("phi_loc", dist.VonMises(pi, 2.0))
            psi_loc = numpyro.sample("psi_loc", dist.VonMises(0.0, 0.1))

            phi_conc = numpyro.sample(
                "phi_conc", dist.Beta(halpha_phi, beta_count_phi - halpha_phi)
            )
            psi_conc = numpyro.sample(
                "psi_conc", dist.Beta(halpha_psi, beta_count_psi - halpha_psi)
            )
            corr_scale = numpyro.sample("corr_scale", dist.Beta(2.0, 10.0))

        with numpyro.plate("obs_plate", num_data, dim=-1):
            assign = numpyro.sample(
                "mix_comp", dist.Categorical(mix_weights), infer={"enumerate": "parallel"}
            )

            phi_psi = numpyro.sample("phi_psi", dist.SineBivariateVonMises(
                phi_loc=phi_loc[assign],
                psi_loc=psi_loc[assign],
                # These concentrations are an order of magnitude lower than expected (550-1000)!
                phi_concentration=70 * phi_conc[assign],
                psi_concentration=70 * psi_conc[assign],
                weighted_correlation=corr_scale[assign],
            ), obs=data)

            return 1

    @staticmethod
    @numpyro.handlers.reparam(
        config={"phi_loc": CircularReparam(), "psi_loc": CircularReparam()}
    )
    def skewed_sine_bivariate_von_mises(data, num_data, n_components=2):
        # Mixture prior
        mix_weights = numpyro.sample("mix_weights", dist.Dirichlet(jnp.ones((n_components,))))

        # Hprior BvM
        # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
        beta_mean_phi = numpyro.sample("beta_mean_phi", dist.Uniform(0.0, 1.0))
        beta_count_phi = numpyro.sample(
            "beta_count_phi", dist.Gamma(1.0, 1.0 / n_components)
        )  # shape, rate
        halpha_phi = beta_mean_phi * beta_count_phi
        beta_mean_psi = numpyro.sample("beta_mean_psi", dist.Uniform(0, 1.0))
        beta_count_psi = numpyro.sample(
            "beta_count_psi", dist.Gamma(1.0, 1.0 / n_components)
        )  # shape, rate
        halpha_psi = beta_mean_psi * beta_count_psi

        with numpyro.plate("mixture", n_components):
            # BvM priors

            # Place gap in forbidden region of the Ramachandran plot (protein backbone dihedral angle pairs)
            phi_loc = numpyro.sample("phi_loc", dist.VonMises(pi, 2.0))
            psi_loc = numpyro.sample("psi_loc", dist.VonMises(0.0, 0.1))

            phi_conc = numpyro.sample(
                "phi_conc", dist.Beta(halpha_phi, beta_count_phi - halpha_phi)
            )
            psi_conc = numpyro.sample(
                "psi_conc", dist.Beta(halpha_psi, beta_count_psi - halpha_psi)
            )
            corr_scale = numpyro.sample("corr_scale", dist.Beta(2.0, 10.0))

            # Skewness prior
            ball_transform = L1BallTransform()
            skewness = numpyro.sample("skewness", dist.Normal(0, 0.5).expand((2,)).to_event(1))
            skewness = ball_transform(skewness)

        with numpyro.plate("obs_plate", num_data, dim=-1):
            assign = numpyro.sample(
                "mix_comp", dist.Categorical(mix_weights), infer={"enumerate": "parallel"}
            )

            sine = dist.SineBivariateVonMises(
                phi_loc=phi_loc[assign],
                psi_loc=psi_loc[assign],
                # These concentrations are an order of magnitude lower than expected (550-1000)!
                phi_concentration=70 * phi_conc[assign],
                psi_concentration=70 * psi_conc[assign],
                weighted_correlation=corr_scale[assign],
            )
            return numpyro.sample("phi_psi", dist.SineSkewed(sine, skewness[assign]), obs=data)



    @staticmethod
    @numpyro.handlers.reparam(
        config={"phi_loc": CircularReparam(), "psi_loc": CircularReparam()}
    )
    def product_von_mises(data, num_data, n_components=2):
        if data is not None:
            phi_data = data[:, 0]
            psi_data = data[:, 1]
        else:
            phi_data = None
            psi_data = None
        # Priors for the mixture weights
        mix_weights = numpyro.sample("mix_weights", dist.Dirichlet(jnp.ones((n_components,))))

        # Hprior BvM
        # Bayesian Inference and Decision Theory by Kathryn Blackmond Laskey
        beta_mean_phi = numpyro.sample("beta_mean_phi", dist.Uniform(0.0, 1.0))
        beta_count_phi = numpyro.sample(
            "beta_count_phi", dist.Gamma(1.0, 1.0 / n_components)
        )  # shape, rate
        halpha_phi = beta_mean_phi * beta_count_phi
        beta_mean_psi = numpyro.sample("beta_mean_psi", dist.Uniform(0, 1.0))
        beta_count_psi = numpyro.sample(
            "beta_count_psi", dist.Gamma(1.0, 1.0 / n_components)
        )  # shape, rate
        halpha_psi = beta_mean_psi * beta_count_psi

        with numpyro.plate("mixture", n_components):
            # BvM priors
            # Place gap in forbidden region of the Ramachandran plot (protein backbone dihedral angle pairs)
            phi_loc = numpyro.sample("phi_loc", dist.VonMises(pi, 2.0))
            psi_loc = numpyro.sample("psi_loc", dist.VonMises(0.0, 0.1))

            phi_conc = numpyro.sample(
                "phi_conc", dist.Beta(halpha_phi, beta_count_phi - halpha_phi)
            )
            psi_conc = numpyro.sample(
                "psi_conc", dist.Beta(halpha_psi, beta_count_psi - halpha_psi)
        )

        # Assign data points to components (latent variable z for each data point)
        with numpyro.plate("obs_plate", num_data, dim=-1):
            assign = numpyro.sample(
                "mix_comp", dist.Categorical(mix_weights), infer={"enumerate": "parallel"}
            )
            phi = numpyro.sample('phi', dist.VonMises(loc=phi_loc[assign], concentration=70*phi_conc[assign]), obs=phi_data)

            # Likelihood for the product of von Mises distributions
            # return numpyro.sample('psi', VonMises(loc=phi[assign]*psi_loc[assign], concentration=70*phi[assign]*psi_conc[assign]), obs=psi_data)
            return numpyro.sample('psi', dist.VonMises(loc=phi * psi_loc[assign], concentration=70*psi_conc[assign]),
                                  obs=psi_data)
