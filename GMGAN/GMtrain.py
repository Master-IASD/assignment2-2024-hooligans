import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import subprocess
from GMmodel import Generator, Discriminator, GaussianM
from GMutils import D_train, G_train, save_models, load_model, generate_fake_samples

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument("--d", type=int, default=100, help="Latent space dimension")

    parser.add_argument("--K", type=int, default=11, help="Number of clusters + 1")

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

    print("Model Loading...")
    mnist_dim = 784
    # K= size of the output of discrimnator
    K = args.K
    d = args.d
    G = Generator(mnist_dim, d)
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

    print("Start Training :")

    n_epoch = args.epochs
    for epoch in trange(1, n_epoch + 1, leave=True):
        n_batch = 0
        dl = 0
        gl = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            dl += D_train(x, y, G, D, GM, D_optimizer, criterion)
            gl += G_train(x, G, D, GM, G_optimizer, GM_optimizer, criterion)
            n_batch += 1
        print(f"Epoch {epoch}, loss D : {dl/n_batch}, lossG : {gl/n_batch}")

        if epoch == 1 or epoch % 25 == 0:
            # Save the checkpoints
            os.makedirs(f"checkpoints_{args.d}_{epoch}", exist_ok=True)
            save_models(G, D, GM, f"checkpoints_{args.d}_{epoch}")
            real_images_path = "real_mnist_png"
            generated_images_path = "samples_train"
            generate_fake_samples(G, GM, 10000)

            # Calculate the FID
            # Call the pytorch-fid script
            fid = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytorch_fid",
                    "real_mnist_png",
                    "samples_train",
                    "--device",
                    "cuda:0",
                ],
                capture_output=True,
                text=True,
            )

            print(f"Epoch {epoch}, FID: {fid.stdout}")

    print("Training done")
