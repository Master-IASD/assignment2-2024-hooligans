# import torch
# import os



# def D_train(x, G, D, D_optimizer, criterion):
#     #=======================Train the discriminator=======================#
#     D.zero_grad()

#     # train discriminator on real
#     x_real, y_real = x, torch.ones(x.shape[0], 1)
#     x_real, y_real = x_real.cuda(), y_real.cuda()

#     D_output = D(x_real)
#     D_real_loss = criterion(D_output, y_real)
#     D_real_score = D_output

#     # train discriminator on facke
#     z = torch.randn(x.shape[0], 100).cuda()
#     x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

#     D_output =  D(x_fake)
    
#     D_fake_loss = criterion(D_output, y_fake)
#     D_fake_score = D_output

#     # gradient backprop & optimize ONLY D's parameters
#     D_loss = D_real_loss + D_fake_loss
#     D_loss.backward()
#     D_optimizer.step()
        
#     return  D_loss.data.item()


# def G_train(x, G, D, G_optimizer, criterion):
#     #=======================Train the generator=======================#
#     G.zero_grad()

#     z = torch.randn(x.shape[0], 100).cuda()
#     y = torch.ones(x.shape[0], 1).cuda()
                 
#     G_output = G(z)
#     D_output = D(G_output)
#     G_loss = criterion(D_output, y)

#     # gradient backprop & optimize ONLY G's parameters
#     G_loss.backward()
#     G_optimizer.step()
        
#     return G_loss.data.item()



# def save_models(G, D, folder):
#     torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
#     torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


# def load_model(G, folder):
#     ckpt = torch.load(os.path.join(folder,'G.pth'))
#     G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
#     return G

import torch
import os






def D_train_WGAN(real_samples, G, D, optimizer, clip_value=0.01):
    # Generate fake samples
    z = torch.randn(real_samples.size(0), 100).cuda()
    fake_samples = G(z)

    # Calculate WGAN loss and gradient penalty
    D_real = D(real_samples)
    D_fake = D(fake_samples.detach())
    
    # Compute the gradient penalty
    gradient_penalty = D.gradient_penalty(real_samples, fake_samples.detach())

    # Combine losses
    D_loss = D_fake.mean() - D_real.mean() + 5 * gradient_penalty
    
    optimizer.zero_grad()
    D_loss.backward()
    optimizer.step()

    # Clip weights
    for p in D.parameters():
        p.data.clamp_(-clip_value, clip_value)

    return D_loss.item()







def G_train_WGAN(G, D, G_optimizer):
    G.zero_grad()

    # Generate fake images
    z = torch.randn(64, 100).cuda()  # Use the batch size directly
    x_fake = G(z)
    D_fake = D(x_fake).mean()

    # Generator loss (Wasserstein)
    G_loss = -D_fake
    G_loss.backward()
    G_optimizer.step()

    return G_loss.item()

def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder, 'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder, 'D.pth'))

def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder, 'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

