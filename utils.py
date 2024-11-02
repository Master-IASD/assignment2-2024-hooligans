import torch
import torchvision
import os
import numpy as np
import torch.nn as nn

d = 100 #dimension of latent space
K = 11 #size of the output of discrimnator


def D_train(x, y, G, D, GaussianM, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    G.train()
    D.train()
    D.zero_grad()


    # train discriminator on real samples
    x_real, y_real = x, y
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output_real = D(x_real)
    D_real_loss = criterion(D_output_real, y_real)

    #representing one of the K Gaussian distributions
    k_values = torch.randint(0, 10, (x.shape[0],))
    y = torch.eye(K)[k_values].cuda()
    N = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))

    #random noise
    z = N.sample((x.shape[0],)).cuda().to(torch.float32)

    #the vector of latent space sampled from the Gaussian Mixture
    z_tilde = GaussianM(y, z)

    #Generate fake sample x_fake
    x_fake = G(z_tilde)

    D_output_fake =  D(x_fake)
    target_fake = torch.full((x.shape[0],), 10, dtype=torch.long).cuda()

    D_fake_loss = criterion(D_output_fake, target_fake)

    # gradient backpropagation and optimization of D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()




def G_train(x, y, G, D, GaussianM, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.train()
    D.train()
    G.zero_grad()



    #representing one of the K Gaussian distributions
    k_values = torch.randint(0, 10, (x.shape[0],))
    y = torch.eye(K)[k_values].cuda()
    N = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
    #random noise
    z = N.sample((x.shape[0],)).cuda().to(torch.float32)

    #the vector of latent space sampled from the Gaussian Mixture
    z_tilde = GaussianM(y, z)

    G_output = G(z_tilde)

    D_output = D(G_output)
    G_loss = criterion(D_output, torch.argmax(y, dim=1)) 

    # gradient backpropagation and optimization of G and GM's parameters
    G_loss.backward()
    G_optimizer.step()
    #GM is an extension of two layers of the generator


    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G
