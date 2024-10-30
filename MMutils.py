import torch
import os
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
#from MMgan import Generator
import torch.nn.functional as F


def train_discriminator(D, G, E, x_r, latent_dim, optimizer_D):

    # Move real data to the specified device
    x_r = x_r.cuda()
    batch_size = x_r.size(0)

    # Step 1: Generate noise and create fake data
    y = torch.argmax(E(x_r), dim=1)  # Cluster labels based on encoder
    z = torch.randn(batch_size, latent_dim).cuda()
    x_f = G(z, y)

    # --------------------------
    # Discriminator Update
    # --------------------------

    # Zero gradients for discriminator
    optimizer_D.zero_grad()

    # Compute discriminator scores for real and fake data
    noise = torch.randn_like(x_r) * 0.1  # Adjust noise level as needed
    d_real = D(x_r + noise)
    d_fake = D(x_f + noise)

    # Compute the difference and apply the sigmoid scaling function
    score_diff = d_real - d_fake
    scaled_diff = torch.sigmoid(score_diff)  # s(C(x_r) - C(x_f))

    # Log-scaled difference for discriminator loss
    log_scaled_diff = torch.log(scaled_diff + 1e-8)  # Adding epsilon to prevent log(0)
    d_loss = -log_scaled_diff.mean()  # Mean across batch

    # Backpropagation and optimization for discriminator
    d_loss.backward()
    optimizer_D.step()

    return d_loss.item()


def train_generator_encoder(
    G, E, D, x_r, num_clusters, latent_dim, alpha, optimizer_G, optimizer_E
):
    # Move real data to the specified device
    x_r = x_r.cuda()
    batch_size = x_r.size(0)

    y = torch.argmax(E(x_r), dim=1)  # Cluster labels based on encoder
    # Sample new cluster labels for fake data
    y_p = torch.randint(0, num_clusters, (batch_size,)).cuda()

    # Generate noise and create fake data conditioned on sampled clusters
    z = torch.randn(batch_size, latent_dim).cuda()
    x_f = G(z, y)
    x_f_p = G(z, y_p)  # Re-generate with sampled cluster labels

    # --------------------------
    # Generator-Encoder Update
    # --------------------------

    # Zero gradients for generator and encoder
    optimizer_G.zero_grad()
    optimizer_E.zero_grad()

    # Compute discriminator scores for real and fake data
    d_real = D(x_r)
    d_fake = D(x_f)

    # Adversarial loss component for generator
    g_e_scaled_diff = torch.sigmoid(d_fake - d_real)  # s(C(x_r) - C(x_f))
    g_e_log_scaled_diff = torch.log(g_e_scaled_diff + 1e-8)  # Numerical stability
    g_loss_adv = -g_e_log_scaled_diff.mean()  # Adversarial component

    # Cluster consistency loss for generated data
    y_pred = E(x_f_p)  # Encoder output for fake data
    p_e = y_pred.gather(1, y_p.view(-1, 1)).squeeze(
        1
    )  # Get probabilities for true labels
    log_true_label_probs = torch.log(p_e + 1e-8)  # Log for stability
    cluster_consistency_loss = -alpha * log_true_label_probs.mean()

    # Total Generator-Encoder loss
    g_e_loss = g_loss_adv + cluster_consistency_loss

    # Backpropagation and optimization for generator and encoder
    g_e_loss.backward()
    optimizer_G.step()
    optimizer_E.step()

    return g_e_loss.item()


def save_models(G, D, E, folder):
    os.makedirs(folder, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(folder, "MMG.pth"))
    torch.save(D.state_dict(), os.path.join(folder, "MMD.pth"))
    torch.save(E.state_dict(), os.path.join(folder, "MME.pth"))


def save_losses(d_l, g_e_l, epoch, folder):
    # Save losses as a .pth file
    torch.save(
        {"epoch": epoch, "MMd_losses": d_l, "MMg_e_losses": g_e_l},
        f"{folder}/losses_epoch_{epoch}.pth",
    )


def load_generator(folder="checkpoints"):
    # Initialize the Generator
    G = Generator()

    # Load the saved state dictionary
    checkpoint = torch.load(os.path.join(folder, "MMG.pth"))

    # Remove 'module.' prefix if saved with DataParallel
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    G.load_state_dict(checkpoint)

    # Move model to CUDA and set to evaluation mode
    G = G.cuda()
    G.eval()

    return G


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_cluster_alignment(
    G, E, num_clusters, latent_dim, real_samples, real_labels, method="tsne"
):
    # Step 1: Generate synthetic samples from each cluster
    generated_samples = []
    generated_labels = []

    for cluster in range(num_clusters):
        z = torch.randn(100, latent_dim).cuda()  # Generate 100 samples for each cluster
        y_cluster = torch.full((100,), cluster).long().cuda()  # Label for the cluster
        gen_samples = G(z, y_cluster).detach().cpu().numpy()
        generated_samples.append(gen_samples)
        generated_labels += [cluster] * 100

    generated_samples = np.vstack(generated_samples)
    generated_labels = np.array(generated_labels)

    # Step 2: Get encoder predictions for generated samples
    gen_labels_encoded = (
        torch.argmax(E(torch.tensor(generated_samples).float().cuda()), dim=1)
        .cpu()
        .numpy()
    )

    # Step 3: Concatenate real and generated samples for visualization
    all_samples = np.vstack([real_samples, generated_samples])
    all_labels = np.concatenate([real_labels, generated_labels])
    all_encoded_labels = np.concatenate([real_labels, gen_labels_encoded])

    # Step 4: Dimensionality reduction for visualization
    if method == "tsne":
        reduced_samples = TSNE(n_components=2).fit_transform(all_samples)
    elif method == "pca":
        reduced_samples = PCA(n_components=2).fit_transform(all_samples)

    # Step 5: Plot real vs generated clusters
    plt.figure(figsize=(12, 6))

    # Plot real samples
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        reduced_samples[: len(real_samples), 0],
        reduced_samples[: len(real_samples), 1],
        c=real_labels,
        cmap="viridis",
        alpha=0.6,
    )
    plt.legend(*scatter.legend_elements(), title="Real Clusters")
    plt.title("Real Samples")

    # Plot generated samples with predicted clusters
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(
        reduced_samples[len(real_samples) :, 0],
        reduced_samples[len(real_samples) :, 1],
        c=gen_labels_encoded,
        cmap="viridis",
        alpha=0.6,
    )
    plt.legend(*scatter.legend_elements(), title="Predicted Clusters")
    plt.title("Generated Samples (Predicted Clusters)")

    plt.show()


from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


def estimate_pdf(samples, bandwidth=0.1, x_range=(-3, 3), num_points=100):
    # Estimate PDF using KDE
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(samples.reshape(-1, 1))
    x_vals = np.linspace(*x_range, num_points).reshape(-1, 1)
    log_pdf = kde.score_samples(x_vals)
    pdf = np.exp(log_pdf)
    return x_vals.flatten(), pdf


def compute_precision_recall(real_pdf, gen_pdf, lambda_values):
    precision = []
    recall = []
    for lam in lambda_values:
        alpha_lambda = np.sum(np.minimum(lam * real_pdf, gen_pdf))
        beta_lambda = np.sum(np.minimum(real_pdf, gen_pdf / lam))
        precision.append(alpha_lambda)
        recall.append(beta_lambda)
    return precision, recall


def plot_pr_curve(
    real_samples, gen_samples, bandwidth=0.1, x_range=(-3, 3), num_points=100
):
    # Step 1: Project real and generated samples to 1D using PCA
    pca = PCA(n_components=1)
    real_samples_1d = pca.fit_transform(real_samples).flatten()
    gen_samples_1d = pca.transform(gen_samples).flatten()

    # Step 2: Estimate PDFs for real and generated samples
    x_vals, real_pdf = estimate_pdf(real_samples_1d, bandwidth, x_range, num_points)
    _, gen_pdf = estimate_pdf(gen_samples_1d, bandwidth, x_range, num_points)

    # Step 3: Define lambda values for PR curve
    lambda_values = np.linspace(0.1, 5, 50)
    precision, recall = compute_precision_recall(real_pdf, gen_pdf, lambda_values)

    # Step 4: Plot PR Curve
    plt.figure(figsize=(12, 6))

    # Plot the PR Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker="o", color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")

    # Plot the PDFs for visual comparison
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, real_pdf, label="Real PDF", color="g")
    plt.plot(x_vals, gen_pdf, label="Generated PDF", color="r")
    plt.xlabel("1D Projected Space")
    plt.ylabel("Density")
    plt.title("PDF Comparison")
    plt.legend()

    plt.show()


import subprocess


def calculate_fid_score(real_image_folder, generated_image_folder):
    """
    Calculate the FID score between real and generated images.

    Parameters:
        real_image_folder (str): Path to the folder containing real images in .png format.
        generated_image_folder (str): Path to the folder containing generated images in .png format.

    Returns:
        None: Prints FID score to the console.
    """
    # Run the FID calculation using pytorch-fid
    subprocess.run(
        ["python", "-m", "pytorch_fid", real_image_folder, generated_image_folder]
    )


# Updated MNIST to PNG function
def save_mnist_as_png(output_folder, train=True):
    os.makedirs(output_folder, exist_ok=True)
    dataset = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    for idx, (img, label) in enumerate(dataset):
        img = img.squeeze(0)
        pil_img = transforms.ToPILImage()(img)
        pil_img.save(f"{output_folder}/image_{idx}.png")

    print(f"Saved {len(dataset)} images to '{output_folder}'.")
