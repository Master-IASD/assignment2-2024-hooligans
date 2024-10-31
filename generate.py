import torch 
import torchvision
import os
import shutil
import argparse


from model import Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="Dimensionality of the latent space.")
    args = parser.parse_args()
    #latent_dim = args.latent_dim
    latent_dim = 200
    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784
    model = Generator(g_output_dim = mnist_dim, latent_dim=latent_dim).cuda()
    model = load_model(model, f'checkpoints/latent_dim_{latent_dim}')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')

    print('Start Generating')
    # Clean up the 'samples' directory if it exists
    if os.path.exists('samples'):
        shutil.rmtree('samples')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, latent_dim).cuda()
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1