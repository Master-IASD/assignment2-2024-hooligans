import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from model import Generator, Discriminator
from utils import D_train_WGAN, G_train_WGAN, save_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WGAN.')
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD")
    parser.add_argument("--n_critic", type=int, default=5, help="Number of D updates per G update")

    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    print('Dataset Loaded.')

    print('Model Loading...')

    # Model Initialization
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim=mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(d_input_dim=mnist_dim)).cuda()
    print('Model loaded.')

    # Optimizers
    G_optimizer = optim.RMSprop(G.parameters(), lr=args.lr)
    D_optimizer = optim.RMSprop(D.parameters(), lr=args.lr)

    # Training Loop
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch + 1, leave=True):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).cuda()

            # Train Discriminator (n_critic times per Generator update)
            for _ in range(args.n_critic):
                D_loss = D_train_WGAN(x, G,D.module, D_optimizer, clip_value=0.01)

            # Train Generator
            G_loss = G_train_WGAN(G, D.module, G_optimizer)  # Pass the current batch size

        # Save model every 10 epochs
        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

    print('Training done')
