import torch
import os
import numpy as np
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Generator, Discriminator,GaussianM
from utils import D_train, G_train, save_models



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Supervised Static GM-GAN.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent space dimension.")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes.")
    parser.add_argument("--K", type=int, default=11, help="Number of Gaussians in mixture.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Covariance scaling factor.")

    args = parser.parse_args()

    
    latent_dim = args.latent_dim

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')

    
    print('Model Loading...')
    mnist_dim = 784
    K = args.K
    d = args.latent_dim
    sigma = args.sigma
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim,K)).cuda()
    

    #initializing gaussian mixture parameters (mu and sigma)


    print('Model loaded.')

    # define loss
    criterion = nn.CrossEntropyLoss()

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)
    GaussianM_instance = GaussianM(K, d).cuda()

   
    

    # model = DataParallel(model).cuda()
    
    # Optimizer 


   

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, args.epochs + 1, leave=True):
        G_loss_total = 0.0
        D_loss_total = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).cuda()
            y = y.cuda()

            D.zero_grad()  # Reset gradients
            D_loss = D_train(x, y, G, D, GaussianM_instance, D_optimizer, criterion)
            D_loss_total += D_loss
            

            # Train the Generator
            G.zero_grad()  # Reset gradients
            G_loss = G_train(x, y, G, D, GaussianM_instance, G_optimizer, criterion)
            G_loss_total += G_loss
            
            if epoch % 10 == 0:
                save_models(G, D, 'checkpoints')

    print('Training done')
