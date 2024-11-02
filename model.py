import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, g_output_dim, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
        self.num_classes = num_classes

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

class Discriminator(nn.Module):
    def __init__(self, d_input_dim, K=11):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, K)
        self.K = K

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return self.fc4(x)


class GaussianM(nn.Module):
    """
    Class that represents the Gaussian Mixture module
    Output follows the k Gaussian Distributions of the latent space
    """
    def __init__(self, K, d, sigma=0.4):
        super(GaussianM, self).__init__()
        self.K = K
        self.d = d
        self.sigma = sigma  # Scaling factor for the covariance matrix

        # Define fixed means and the standard deviation scaling factor
        self.means = torch.zeros(K, d).cuda()  # Fixed mean vectors for each Gaussian component
        self.sigma_matrix = self.sigma * torch.eye(d).cuda()  # Covariance matrix: sigma * I_d

    def forward(self, k, z):
        # Get the mean for the selected components
        mu = self.means[k.argmax(dim=1)]  # Get the mean vector corresponding to each k

        # Compute the sampled vector from the Gaussian mixture
        # z is assumed to be standard normal noise
        return mu + (self.sigma_matrix @ z.unsqueeze(-1)).squeeze(-1)
