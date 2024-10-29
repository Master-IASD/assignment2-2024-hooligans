import torch
import torchvision
import os
import argparse
import utils
from MMgan import Generator
from MMutils import load_generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples with MMGAN.")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for generation."
    )
    parser.add_argument(
        "--z_dim", type=int, default=30, help="Dimension of the latent space."
    )
    parser.add_argument(
        "--num_clusters", type=int, default=10, help="Number of clusters."
    )
    args = parser.parse_args()

    print("Loading model...")
    MMG = load_generator()
    print("Model loaded.")

    print("Generating samples...")
    os.makedirs("MMsamples", exist_ok=True)
    n_samples = 10000

    num_clusters = 10
    samples_per_cluster = int(n_samples / num_clusters)
    latent_dim = 100

    with torch.no_grad():
        for cluster in range(num_clusters):
            # Generate 1000 samples for the current cluster
            z = torch.randn(samples_per_cluster, latent_dim).cuda()  # Latent vectors
            cluster_labels = torch.full(
                (samples_per_cluster,), cluster, dtype=torch.long
            ).cuda()  # Cluster labels

            # Generate samples
            x = MMG(z, cluster_labels).reshape(samples_per_cluster, 28, 28)

            # Save each sample
            for k in range(samples_per_cluster):
                # Compute unique index for each sample
                sample_index = cluster * samples_per_cluster + k
                torchvision.utils.save_image(
                    x[k : k + 1],  # Single sample
                    os.path.join("MMsamples", f"{sample_index}.png"),
                )

    print(f"Generated {n_samples} samples.")
