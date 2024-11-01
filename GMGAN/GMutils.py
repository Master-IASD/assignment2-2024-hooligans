import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = 100  # dimension of latent space
K = 11  # size of the output of discrimnator


def D_train(x, y, G, D, GaussianM, D_optimizer, criterion):
    # =======================Train the discriminator=======================#
    G.train()
    D.train()
    GaussianM.train()
    D.zero_grad()

    # train discriminator on real samples
    x_real, y_real = x, y
    x_real, y_real = x_real.to(DEVICE), y_real.to(DEVICE)

    D_output_real = D(x_real)
    D_real_loss = criterion(D_output_real, y_real)

    # representing one of the K Gaussian distributions
    k_values = torch.randint(0, 10, (x.shape[0],))
    y = torch.eye(K)[k_values].to(DEVICE)
    N = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))

    # random noise
    z = N.sample((x.shape[0],)).to(DEVICE).to(torch.float32)

    # the vector of latent space sampled from the Gaussian Mixture
    z_tilde = GaussianM(y, z)

    # Generate fake sample x_fake
    x_fake = G(z_tilde)

    D_output_fake = D(x_fake)
    target_fake = torch.full((x.shape[0],), 10, dtype=torch.long).to(DEVICE)

    D_fake_loss = criterion(D_output_fake, target_fake)

    # gradient backpropagation and optimization of D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(
    x, y, G, D, GaussianM, G_optimizer, GM_optimizer, criterion, n_batch, epoch
):
    # =======================Train the generator=======================#
    G.train()
    D.train()
    GaussianM.train()
    G.zero_grad()
    GaussianM.zero_grad()

    # representing one of the K Gaussian distributions
    k_values = torch.randint(0, 10, (x.shape[0],))
    y = torch.eye(K)[k_values].to(DEVICE)
    N = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
    # random noise
    z = N.sample((x.shape[0],)).to(DEVICE).to(torch.float32)

    # the vector of latent space sampled from the Gaussian Mixture
    z_tilde = GaussianM(y, z)

    G_output = G(z_tilde)

    D_output = D(G_output)
    G_loss = criterion(D_output, torch.argmax(y, dim=1))  # le vrai y

    # gradient backpropagation and optimization of G and GM's parameters
    G_loss.backward()
    G_optimizer.step()
    # GM is an extension of two layers of the generator
    GM_optimizer.step()

    return G_loss.data.item()


def save_models(G, D, GaussianM, folder):
    torch.save(G.state_dict(), os.path.join(folder, "G.pth"))
    torch.save(D.state_dict(), os.path.join(folder, "D.pth"))
    torch.save(GaussianM.state_dict(), os.path.join(folder, "GaussMixt.pth"))


def load_model(G, GaussianM, folder, Discriminator=None):
    ckpt_G = torch.load(os.path.join(folder, "G.pth"))
    ckpt_GM = torch.load(os.path.join(folder, "GaussMixt.pth"))
    G.load_state_dict({k.replace("module.", ""): v for k, v in ckpt_G.items()})
    GaussianM.load_state_dict({k.replace("module.", ""): v for k, v in ckpt_GM.items()})
    if not Discriminator == None:
        ckpt_D = torch.load(os.path.join(folder, "D.pth"))
        Discriminator.load_state_dict(
            {k.replace("module.", ""): v for k, v in ckpt_D.items()}
        )
