import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, g_output_dim, d):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(d, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    """
    Class that represent the Discriminator module
    Output a vector of size 11
    """

    def __init__(self, d_input_dim, K, dropprob=0.3):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, K)

        # Adding dropout
        self.drop1 = nn.Dropout(p=dropprob)
        self.drop2 = nn.Dropout(p=dropprob)
        self.drop3 = nn.Dropout(p=dropprob)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.drop1(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.drop2(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.drop3(x)
        return self.fc4(x)


class GaussianM(nn.Module):
    """
    Class that represent the Gaussian Mixture module
    Output follows the k Gaussian Distribution of the latent space
    """

    def __init__(self, K, d):
        super(GaussianM, self).__init__()
        self.fcmu = nn.Linear(K, d)
        self.fcsigma = nn.Linear(K, d)

    def forward(self, k, z):
        mu = self.fcmu(k)
        sigma = self.fcsigma(k)
        return mu + (sigma * z)
