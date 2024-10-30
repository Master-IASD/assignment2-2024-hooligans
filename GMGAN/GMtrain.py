import torch
import torchvision
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pickle
from pytorch_fid import fid_score
from pytorch_fid.fid_score import calculate_fid_given_paths
import matplotlib.pyplot as plt
from GMmodel import Generator, Discriminator, GaussianM
from GMutils import D_train, G_train, save_models, load_model
from torchvision.utils import save_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_real_samples(train_loader):
    """Function to save real samples of MNIST, used to calculate FID"""

    real_images_dir = "data/MNIST_raw"
    os.makedirs(real_images_dir, exist_ok=True)
    for batch_idx, (x, _) in enumerate(train_loader):
        if x.shape[0] != args.batch_size:
            image = x.reshape(x.shape[0], 28, 28)
        else:
            image = x.reshape(args.batch_size, 28, 28)
        for k in range(x.shape[0]):
            filename = os.path.join(
                real_images_dir, f"real_image_{batch_idx * args.batch_size + k}.png"
            )
            save_image(image[k : k + 1], filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Normalizing Flow.")
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs for training."
    )
    parser.add_argument(
        "--lr", type=float, default=8e-5, help="The learning rate to use for training."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Size of mini-batches for SGD"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Load a pre trained model to fine tune"
    )

    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("samples_train", exist_ok=True)

    # Data Pipeline
    print("Dataset loading...")
    # MNIST Dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5))]
    )

    train_dataset = datasets.MNIST(
        root="data/MNIST/", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="data/MNIST/", train=False, transform=transform, download=False
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )
    print("Dataset Loaded.")

    if not (os.path.exists("data/MNIST_raw")):
        print("Saving test set locally ...")
        save_real_samples(test_loader)

    print("Model Loading...")
    mnist_dim = 784
    # K= size of the output of discrimnator
    K = 11
    d = 100
    G = Generator(g_output_dim=mnist_dim)
    D = Discriminator(mnist_dim, K)
    GM = GaussianM(K, d)
    if args.model:
        load_model(G, GM, args.model, Discriminator=D)
    G = torch.nn.DataParallel(G).to(DEVICE)
    D = torch.nn.DataParallel(D).to(DEVICE)
    GM = torch.nn.DataParallel(GM).to(DEVICE)

    # initializing gaussian mixture parameters (mu and sigma)
    sigma_init = 1.4
    c = 3
    for name, param in GM.named_parameters():
        if "fcsigma.weight" in name:
            nn.init.constant_(param, sigma_init)
        if "fcmu.weight" in name:
            nn.init.uniform_(param, -c, c)

    print("Model loaded.")

    # define loss
    criterion = nn.CrossEntropyLoss()

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    GM_optimizer = optim.Adam(GM.parameters(), lr=1e-9, betas=(0.5, 0.999))

    def generate_fake_samples(generator, gm, num_samples):
        """Function to generate fake samples using the generator"""
        n_samples = 0
        with torch.no_grad():
            while n_samples < num_samples:
                z = torch.randn(args.batch_size, 100).to(DEVICE)
                k_values = torch.randint(0, 10, (args.batch_size,))
                y = torch.eye(K)[k_values].to(DEVICE)
                N = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
                z = N.sample((args.batch_size,)).to(DEVICE).to(torch.float32)
                z_tilde = gm(y, z)
                x = generator(z_tilde)
                x = x.reshape(args.batch_size, 28, 28)
                for k in range(x.shape[0]):
                    if n_samples < num_samples:
                        torchvision.utils.save_image(
                            x[k : k + 1],
                            os.path.join("samples_train", f"{n_samples}.png"),
                        )
                        n_samples += 1

    print("Start Training :")

    n_epoch = args.epochs
    fid_values = []
    D_loss = []
    G_loss = []
    for epoch in trange(1, n_epoch + 1, leave=True):
        n_batch = 0
        dl = 0
        gl = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            dl += D_train(x, y, G, D, GM, D_optimizer, criterion)
            gl += G_train(
                x, y, G, D, GM, G_optimizer, GM_optimizer, criterion, n_batch, epoch
            )
            n_batch += 1
        print(f"Epoch {epoch}, loss D : {dl/n_batch}, lossG : {gl/n_batch}")
        G_loss.append(gl / n_batch)
        D_loss.append(dl / n_batch)
        if epoch % 25 == 0:
            # Save the checkpoints
            os.makedirs(f"checkpoints{epoch}", exist_ok=True)
            save_models(G, D, GM, f"checkpoints{epoch}")
            real_images_path = "data/MNIST_raw"
            generated_images_path = "samples_train"
            generate_fake_samples(G, GM, 10000)

            # Calculate the FID
            fid_value = calculate_fid_given_paths(
                [real_images_path, generated_images_path],
                batch_size=args.batch_size,
                device=DEVICE,
                dims=2048,
            )
            print(f"Epoch {epoch}, FID: {fid_value:.2f}")
            fid_values.append(fid_value)

    # Plot the FID
    fig, ax = plt.subplots()
    ax.plot(fid_values, marker="o", linestyle="-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID")
    ax.set_title("FID Over Epochs")
    plt.savefig("fid_plot.png")

    # Plot the loss of the generator over epoch
    fig, ax = plt.subplots()
    ax.plot(G_loss, marker="o", linestyle="-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generator Loss")
    ax.set_title("Generator Loss over training")
    plt.savefig("genloss.png")

    # Plot the loss of the discrimator over epoch
    fig, ax = plt.subplots()
    ax.plot(D_loss, marker="o", linestyle="-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Discriminator Loss")
    ax.set_title("Discriminator Loss over training")
    plt.savefig("disloss.png")

    print("Training done")
