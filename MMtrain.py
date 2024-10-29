import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from MMgan import Generator, Discriminator, Encoder
from MMutils import (
    save_models,
    train_discriminator,
    train_generator_encoder,
    save_losses,
)  # Import the save_models utility

if __name__ == "__main__":
    # Argument parsing for training configuration
    parser = argparse.ArgumentParser(description="Train MMGAN.")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs for training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="The learning rate to use for training.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of mini-batches for SGD"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="Dimension of the latent space"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=10, help="Number of clusters for Encoder"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weighting factor for cluster consistency loss",
    )

    args = parser.parse_args()

    # Setup directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Data Pipeline
    print("Dataset loading...")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
    )
    train_dataset = datasets.MNIST(
        root="data/MNIST/", train=True, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    print("Dataset Loaded.")

    # Model Loading
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator()).cuda()
    D = torch.nn.DataParallel(Discriminator()).cuda()
    E = torch.nn.DataParallel(Encoder()).cuda()

    print("Model loaded.")

    # Define optimizers

    G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.99))
    D_optimizer = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.99))
    E_optimizer = optim.Adam(E.parameters(), lr=0.0002, betas=(0.5, 0.99))

    print("Start Training:")

    # Training loop
    d_loss_list = []
    g_e_loss_list = []
    for epoch in trange(1, args.epochs + 1, leave=True):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).cuda()  # Flatten input and move to CUDA

            # Perform a single training step for the Discriminator
            d_loss = train_discriminator(
                D=D,
                G=G,
                E=E,
                x_r=x,
                latent_dim=args.latent_dim,
                optimizer_D=D_optimizer,
            )

            # Perform a single training step for the Generator and Encoder
            g_e_loss = train_generator_encoder(
                G=G,
                E=E,
                D=D,
                x_r=x,
                num_clusters=args.num_clusters,
                latent_dim=args.latent_dim,
                alpha=args.alpha,
                optimizer_G=G_optimizer,
                optimizer_E=E_optimizer,
            )
        print(
            f"Epoch {epoch}/{args.epochs}, D Loss: {d_loss:.4f}, G+E Loss: {g_e_loss:.4f}"
        )
        # Print and save losses every 10 epochs
        if epoch % 10 == 0:
            save_models(
                G, D, E, "checkpoints"
            )  # Save models to the specified directory
            save_losses(d_loss_list, g_e_loss_list, epoch, "losses")

    print("Training done")
