import torch 
from torch.optim import Adam

import pyro
pyro.enable_validation(True)
pyro.get_param_store().clear() 

from pyro.contrib.gp.kernels import RBF, Periodic, Kernel
from pyro.contrib.gp.models import GPRegression
from pyro.infer.mcmc import NUTS, MCMC
from pyro.infer.svi import SVI
import pyro.distributions as dist
from pyro import poutine

#from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.spatial
import scipy.optimize as opt


# Generate the data
def generate_data(n_points=30, noise_std=0.1):
    x = torch.linspace(0, 1, n_points).unsqueeze(-1)
    g = lambda x: -(torch.sin(6 * torch.pi * x) ** 2) + 6 * x**2 - 5 * x**4 + 1.5
    y_true = g(x)
    epsilon = torch.normal(mean=0.0, std=noise_std, size=y_true.shape)
    y_obs = y_true + epsilon
    return x, y_obs, y_true

# Combined kernel class
class CombinedKernel:
    def __init__(self, input_dim, ls_rbf, var_rbf, ls_periodic, var_periodic, period):
        self.kernel_rbf = RBF(input_dim=input_dim, variance=var_rbf, lengthscale=ls_rbf)
        self.kernel_periodic = Periodic(
            input_dim=input_dim, variance=var_periodic, lengthscale=ls_periodic, period=period
        )
    
    def forward(self, X1, X2):
        return self.kernel_rbf.forward(X1, X2) + self.kernel_periodic.forward(X1, X2)

# Define the GP model
class GPRegression:
    def __init__(self, x_train, y_train, kernel, noise):
        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel
        self.noise = noise
    
    def mean_function(self, x):
        return torch.zeros(x.shape[0])

# Log joint probability
def log_joint(y_obs, x, kernel, gp_model, ls_rbf, var_rbf, ls_periodic, var_periodic):
    
    # Compute kernel matrix
    kernel_matrix = kernel.forward(x, x) + torch.eye(len(x)) * gp_model.noise

    # Log likelihood
    log_likelihood = dist.MultivariateNormal(
        loc=gp_model.mean_function(x), covariance_matrix=kernel_matrix
    ).log_prob(y_obs)

    # Log priors
    log_prior = (
        dist.Gamma(2.0, 1.0).log_prob(ls_rbf) +
        dist.LogNormal(0.0, 1.0).log_prob(var_rbf) +
        dist.Gamma(2.0, 2.0).log_prob(ls_periodic) +
        dist.LogNormal(0.0, 1.0).log_prob(var_periodic)
    )

    return log_likelihood + log_prior



# Compute posterior likelihood
def compute_posterior_likelihood(posterior_samples, x_test, y_test, gp_model):
    
    log_likelihoods = []

    # Iterate through posterior samples
    for i in range(len(posterior_samples["ls_rbf"])):
        # Extract sampled parameters
        sampled_ls_rbf = posterior_samples["ls_rbf"][i].detach()
        sampled_var_rbf = posterior_samples["var_rbf"][i].detach()
        sampled_ls_periodic = posterior_samples["ls_periodic"][i].detach()
        sampled_var_periodic = posterior_samples["var_periodic"][i].detach()
        
        # Create kernel for this sample
        kernel = CombinedKernel(
            input_dim=1,
            ls_rbf=sampled_ls_rbf,
            var_rbf=sampled_var_rbf,
            ls_periodic=sampled_ls_periodic,
            var_periodic=sampled_var_periodic,
            period=torch.tensor(1 / 6)
        )

        # Compute kernel matrix for test set
        kernel_matrix = kernel.forward(x_test, x_test) + torch.eye(len(x_test)) * gp_model.noise

        # Compute log likelihood
        log_likelihood = dist.MultivariateNormal(
            loc=gp_model.mean_function(x_test), covariance_matrix=kernel_matrix
        ).log_prob(y_test)
        log_likelihoods.append(log_likelihood.item())

    # Return mean log likelihood
    return torch.tensor(log_likelihoods).mean().item()


def run_experiment(num_datasets=20, num_samples=500, warmup_steps=100):
    test_likelihoods_map = []
    test_likelihoods_nuts = []

    for i in range(num_datasets):
        print(f"Dataset {i + 1}/{num_datasets}")

        # Generate data
        x, y_obs, y_true = generate_data()
        x_train, x_test = x[:20], x[20:]
        y_train, y_test = y_obs[:20].squeeze(-1), y_obs[20:].squeeze(-1)

        # Reset GP model for this dataset
        kernel = CombinedKernel(
            input_dim = 1,
            ls_rbf = torch.tensor(0.2),
            var_rbf = torch.tensor(1.0),
            ls_periodic = torch.tensor(0.1),
            var_periodic = torch.tensor(0.5),
            period = torch.tensor(1 / 6)
        )
        gp_model = GPRegression(x_train, y_train, kernel, noise=torch.tensor(0.1))

        # ---- Step 1: Fit using MAP ----
        ls_rbf = torch.tensor(0.2, requires_grad=True)
        var_rbf = torch.tensor(1.0, requires_grad=True)
        ls_periodic = torch.tensor(0.1, requires_grad=True)
        var_periodic = torch.tensor(0.5, requires_grad=True)
        optimizer = torch.optim.Adam([ls_rbf, var_rbf, ls_periodic, var_periodic], lr=0.01)
        for _ in range(300):
            optimizer.zero_grad()
            loss = -log_joint(y_train, x_train, kernel, gp_model, ls_rbf, var_rbf, ls_periodic, var_periodic)
            loss.backward()
            optimizer.step()

        # Compute test likelihood for MAP
        kernel = CombinedKernel(
            input_dim=1,
            ls_rbf=ls_rbf.detach(),
            var_rbf=var_rbf.detach(),
            ls_periodic=ls_periodic.detach(),
            var_periodic=var_periodic.detach(),
            period=torch.tensor(1 / 6)
        )
        kernel_matrix = kernel.forward(x_test, x_test) + torch.eye(len(x_test)) * gp_model.noise
        log_likelihood_map = dist.MultivariateNormal(
            loc=gp_model.mean_function(x_test), covariance_matrix=kernel_matrix
        ).log_prob(y_test).item()
        test_likelihoods_map.append(log_likelihood_map)

        # ---- Step 2: Fit using NUTS ----
        def model():
            ls_rbf = pyro.sample("ls_rbf", dist.Gamma(2.0, 1.0))
            var_rbf = pyro.sample("var_rbf", dist.LogNormal(0.0, 1.0))
            ls_periodic = pyro.sample("ls_periodic", dist.Gamma(2.0, 2.0))
            var_periodic = pyro.sample("var_periodic", dist.LogNormal(0.0, 1.0))

            kernel = CombinedKernel(
                input_dim=1,
                ls_rbf=ls_rbf,
                var_rbf=var_rbf,
                ls_periodic=ls_periodic,
                var_periodic=var_periodic,
                period=torch.tensor(1 / 6))

            kernel_matrix = kernel.forward(x_train, x_train) + torch.eye(len(x_train)) * gp_model.noise
            pyro.sample("y_train", dist.MultivariateNormal(
                loc=gp_model.mean_function(x_train), covariance_matrix=kernel_matrix), obs=y_train)
        
        nuts_kernel = NUTS(model, jit_compile = True)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)
        mcmc.run()
        posterior_samples = mcmc.get_samples()

        # Compute test likelihood for NUTS
        log_likelihood_nuts = compute_posterior_likelihood(posterior_samples, x_test, y_test, gp_model)
        test_likelihoods_nuts.append(log_likelihood_nuts)

    # Summary statistics
    results = {
        "MAP": {
            "mean": np.mean(test_likelihoods_map),
            "std": np.std(test_likelihoods_map),
            "all": test_likelihoods_map,
        },
        "NUTS": {
            "mean": np.mean(test_likelihoods_nuts),
            "std": np.std(test_likelihoods_nuts),
            "all": test_likelihoods_nuts,
        },
    }
    return results

# Run the experiment
results = run_experiment(num_datasets=20)
print("Results Summary:")
print(f"MAP: Mean = {results['MAP']['mean']}, Std = {results['MAP']['std']}")
print(f"NUTS: Mean = {results['NUTS']['mean']}, Std = {results['NUTS']['std']}")
