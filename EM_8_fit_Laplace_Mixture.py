import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
from scipy.stats import laplace
import pickle

class LaplaceMixture:
    def __init__(self, n_components, max_iter=100, tol=1e-3, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)
        
    def fit(self, X):
        n_samples = X.shape[0]
        X_flat = X.flatten()
        # Initialize the parameters
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = self.random_state.choice(X_flat, self.n_components)
        self.scales_ = np.ones(self.n_components)
        
        log_likelihood = -np.inf
        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                responsibilities[:, k] = self.weights_[k] * laplace.pdf(
                    X_flat, loc=self.means_[k], scale=self.scales_[k]
                )
            sum_responsibilities = np.sum(responsibilities, axis=1, keepdims=True)
            # Avoid division by zero
            sum_responsibilities[sum_responsibilities == 0] = np.finfo(float).eps
            responsibilities /= sum_responsibilities
            
            # M-step: update parameters
            weights_new = np.sum(responsibilities, axis=0) / n_samples
            means_new = np.zeros(self.n_components)
            scales_new = np.zeros(self.n_components)
            
            for k in range(self.n_components):
                # Update means: weighted median
                means_new[k] = self._weighted_median(X_flat, responsibilities[:, k])
                # Update scales
                scales_new[k] = np.sum(responsibilities[:, k] * np.abs(X_flat - means_new[k])) / np.sum(responsibilities[:, k])
                # Ensure scales are positive
                scales_new[k] = max(scales_new[k], np.finfo(float).eps)
            
            # Compute log likelihood
            log_likelihood_new = np.sum(np.log(sum_responsibilities))
            
            # Check convergence
            if np.abs(log_likelihood_new - log_likelihood) < self.tol:
                break
            log_likelihood = log_likelihood_new
            self.weights_ = weights_new
            self.means_ = means_new
            self.scales_ = scales_new
        return self
    
    def _weighted_median(self, data, weights):
        """Compute the weighted median."""
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumulative_weight = np.cumsum(sorted_weights)
        cutoff = cumulative_weight[-1] / 2.0
        idx = np.searchsorted(cumulative_weight, cutoff)
        return sorted_data[idx]
    
    def predict_proba(self, X):
        """Compute the posterior probabilities (responsibilities)."""
        n_samples = X.shape[0]
        X_flat = X.flatten()
        responsibilities = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * laplace.pdf(
                X_flat, loc=self.means_[k], scale=self.scales_[k]
            )
        sum_responsibilities = np.sum(responsibilities, axis=1, keepdims=True)
        # Avoid division by zero
        sum_responsibilities[sum_responsibilities == 0] = np.finfo(float).eps
        responsibilities /= sum_responsibilities
        return responsibilities
    
    def sample(self, n_samples):
        """Generate samples from the fitted Laplace Mixture Model."""
        samples = np.zeros(n_samples)
        component_choices = self.random_state.choice(
            self.n_components, size=n_samples, p=self.weights_
        )
        for k in range(self.n_components):
            n_k = np.sum(component_choices == k)
            if n_k > 0:
                samples_k = laplace.rvs(
                    loc=self.means_[k], scale=self.scales_[k], size=n_k, random_state=self.random_state
                )
                samples[component_choices == k] = samples_k
        return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit Laplace Mixture Models for each dimension.')
    parser.add_argument("--n_components", type=int, default=10,
                        help="The number of components in each Laplace mixture model.")
    args = parser.parse_args()

    latent_dim = 150  # Ensure this matches your generator's latent dimension

    # Step 1: Load the Data
    data_dir = 'z_real_gen'
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    data_list = []

    for filename in file_list:
        filepath = os.path.join(data_dir, filename)
        tensor = torch.load(filepath)
        tensor = tensor.view(-1, latent_dim).numpy()  # Reshape to (n_samples, latent_dim)
        data_list.append(tensor)

    data_array = np.vstack(data_list)  # Shape: (total_samples, latent_dim)

    # Step 2: Standardize the Data
    scaler = StandardScaler()
    data_array_std = scaler.fit_transform(data_array)

    # Step 3: Fit Laplace Mixture Models for Each Dimension
    laplace_mixtures = []
    for dim in range(latent_dim):
        print(f"Fitting Laplace Mixture for dimension {dim+1}/{latent_dim}")
        lm = LaplaceMixture(n_components=args.n_components, random_state=42)
        lm.fit(data_array_std[:, dim])
        laplace_mixtures.append(lm)

    # Step 4: Save the Models and Scaler
    with open('laplace_mixtures.pkl', 'wb') as f:
        pickle.dump(laplace_mixtures, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Laplace mixture models and scaler saved.")