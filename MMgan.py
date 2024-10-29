import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_clusters=10, output_dim=784):
        super(Generator, self).__init__()
        # Embedding layers for cluster-specific mean and standard deviation
        self.mu_embedding = nn.Embedding(num_clusters, latent_dim)
        self.sigma_embedding = nn.Embedding(num_clusters, 1)

        # Initialize embeddings for mu and sigma with no gradient tracking
        with torch.no_grad():
            nn.init.uniform_(
                self.mu_embedding.weight, -1, 1
            )  # Mean values within [-1, 1]
            self.sigma_embedding.weight.uniform_(
                1, 2
            )  # Standard deviations in range [1, 2]

        # Layers for data generation
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, output_dim)

    def forward(self, z, y_idx):
        mu = self.mu_embedding(y_idx)
        sigma = self.sigma_embedding(y_idx)
        z = mu + sigma * z
        # Pass through generator network
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()

        # Define layers with Batch Normalization after each hidden layer
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.bn2 = nn.BatchNorm1d(self.fc2.out_features)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.bn3 = nn.BatchNorm1d(self.fc3.out_features)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

        # Apply Xavier initialization to each layer
        # self._initialize_weights()

    def forward(self, x):
        # Apply fully connected layers with LeakyReLU and Batch Normalization
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.2)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.2)
        x = self.dropout1(x)
        # Output layer with sigmoid activation
        return torch.sigmoid(self.fc4(x))

    def _initialize_weights(self):
        # Apply Xavier initialization to all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    def __init__(self, input_dim=784, num_clusters=10):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features // 2)
        self.fc5 = nn.Linear(self.fc4.out_features, num_clusters)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.fc4(x), 0.2)
        return F.softmax(self.fc5(x), dim=1)  # Output a distribution over clusters
