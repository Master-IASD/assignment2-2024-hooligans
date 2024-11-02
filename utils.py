import torch
import os

def D_train(x, G, D, D_optimizer, criterion, latent_dim=200):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], latent_dim).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

def G_train(x, G, D, G_optimizer, criterion, latent_dim=200):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], latent_dim).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def save_models(G, D, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(G.module.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.module.state_dict(), os.path.join(folder,'D.pth'))

# def load_model(G, folder):
#     ckpt = torch.load(os.path.join(folder,'G.pth'))
#     G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
#     return G

def load_model(model, folder):
    model.load_state_dict(torch.load(os.path.join(folder, 'G.pth')))
    return model


def GM_trick_batch(batch_size, K, d, sigma, c):
    alpha = 1 / K
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Sigma = sigma * torch.eye(d, device=device)  # (d, d)
    mu = torch.empty(K, d, device=device).uniform_(-c, c)  # (K, d)
    mvn = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
    z = mvn.sample((batch_size,))  # z.shape = (batch_size, K, d)
    z = alpha * z.sum(dim=1)  # Sum over K components
    return z  # Shape: (batch_size, d)

def G_train_fixed_GM(x, G, D, G_optimizer, criterion, K, sigma, c, latent_dim=200):
    G.zero_grad()
    z = GM_trick_batch(x.shape[0], K, latent_dim, sigma, c)
    y = torch.ones(x.shape[0], 1, device=z.device)
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)
    G_loss.backward()
    G_optimizer.step()
    return G_loss.item()