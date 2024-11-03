import torch
import os
import torchvision
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = 100  # dimension of latent space
K = 11  # size of the output of discrimnator


def D_train(x, y, G, D, GM, D_optimizer, criterion):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real samples
    x_real, y_real = x, y
    x_real, y_real = x_real.to(DEVICE), y_real.to(DEVICE)

    D_output_real = D(x_real)
    D_real_loss = criterion(D_output_real, y_real)

    k_values = torch.randint(0, K, (x.shape[0],), device=DEVICE)
    y = F.one_hot(k_values, num_classes=K).to(DEVICE).float()
    z = torch.randn(x.shape[0], d, device=DEVICE, dtype=torch.float32)

    # the vector of latent space sampled from the Gaussian Mixture
    z_tilde = GM(y, z)

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


def G_train(x, G, D, GM, G_optimizer, GM_optimizer, criterion):
    # =======================Train the generator=======================#
    G.zero_grad()
    GM.zero_grad()

    # representing one of the K Gaussian distributions
    k_values = torch.randint(0, K, (x.shape[0],), device=DEVICE)
    y = F.one_hot(k_values, num_classes=K).to(DEVICE).float()
    z = torch.randn(x.shape[0], d, device=DEVICE, dtype=torch.float32)

    # the vector of latent space sampled from the Gaussian Mixture
    z_tilde = GM(y, z)
    G_output = G(z_tilde)
    D_output = D(G_output)

    G_loss = criterion(D_output, k_values)

    # gradient backpropagation and optimization of G and GM's parameters
    G_loss.backward()
    G_optimizer.step()
    # GM is an extension of two layers of the generator
    GM_optimizer.step()

    return G_loss.data.item()


def save_models(G, D, GM, folder):
    torch.save(G.state_dict(), os.path.join(folder, "G.pth"))
    torch.save(D.state_dict(), os.path.join(folder, "D.pth"))
    torch.save(GM.state_dict(), os.path.join(folder, "GM.pth"))


def load_model(G, GM, folder, Discriminator=None):
    ckpt_G = torch.load(os.path.join(folder, "G.pth"))
    ckpt_GM = torch.load(os.path.join(folder, "GM.pth"))
    G.load_state_dict({k.replace("module.", ""): v for k, v in ckpt_G.items()})
    GM.load_state_dict({k.replace("module.", ""): v for k, v in ckpt_GM.items()})
    if not Discriminator == None:
        ckpt_D = torch.load(os.path.join(folder, "D.pth"))
        Discriminator.load_state_dict(
            {k.replace("module.", ""): v for k, v in ckpt_D.items()}
        )


# Updated MNIST to PNG function
def save_mnist_as_png(output_folder, train=True):
    os.makedirs(output_folder, exist_ok=True)
    dataset = torchvision.datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )

    for idx, (img, label) in enumerate(dataset):
        img = img.squeeze(0)
        pil_img = torchvision.transforms.ToPILImage()(img)
        pil_img.save(f"{output_folder}/image_{idx}.png")

    print(f"Saved {len(dataset)} images to '{output_folder}'.")


def generate_fake_samples(generator, gm, num_samples, batch_size=2048):
    """Function to generate fake samples using the generator"""
    n_samples = 0
    with torch.no_grad():
        while n_samples < num_samples:
            z = torch.randn(batch_size, 100).to(DEVICE)
            k_values = torch.randint(0, 10, (batch_size,))
            y = torch.eye(K)[k_values].to(DEVICE)
            N = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
            z = N.sample((batch_size,)).to(DEVICE).to(torch.float32)
            z_tilde = gm(y, z)
            x = generator(z_tilde)
            x = x.reshape(batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples < num_samples:
                    torchvision.utils.save_image(
                        x[k : k + 1],
                        os.path.join("samples_train", f"{n_samples}.png"),
                    )
                    n_samples += 1


if __name__ == "__main__":
    save_mnist_as_png("real_samples")
