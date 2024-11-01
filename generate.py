import torch
import torchvision
import os
import argparse
from GMmodel import Generator, GaussianM
from GMutils import load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Normalizing Flow.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="The batch size to use for training.",
    )
    parser.add_argument("--model", type=str, default="checkpoints200")

    parser.add_argument("--d", type=int, default=100, help="latent space dimension")

    args = parser.parse_args()

    print("Model Loading...")
    # Model Pipeline
    mnist_dim = 784
    K = 11
    d = args.d

    G = Generator(g_output_dim=mnist_dim, d=d).to(DEVICE)
    GM = GaussianM(K, d).to(DEVICE)
    load_model(G, GM, args.model)
    G = torch.nn.DataParallel(G).to(DEVICE)
    G.eval()
    GM = torch.nn.DataParallel(GM).to(DEVICE)
    GM.eval()

    print("Model loaded.")

    print("Start Generating")
    os.makedirs("samples", exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples < 10000:
            k_values = torch.randint(0, 10, (args.batch_size,))
            y = torch.eye(K)[k_values].to(DEVICE)
            N = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
            z = N.sample((args.batch_size,)).to(DEVICE).to(torch.float32)
            z_tilde = GM(y, z)
            x = G(z_tilde)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples < 10000:
                    torchvision.utils.save_image(
                        x[k : k + 1], os.path.join("samples", f"{n_samples}.png")
                    )
                    n_samples += 1
