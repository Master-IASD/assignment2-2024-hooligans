import torch 
import torchvision
import os
import shutil
import argparse
import pickle
import numpy as np

from model import Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Images using Laplace Mixture Models.')
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="The batch size to use for generation.")
    args = parser.parse_args()

    latent_dim = 150  # Ensure this matches your generator's latent dimension

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784
    model = Generator(g_output_dim=mnist_dim, latent_dim=latent_dim)
    model = load_model(model, f'checkpoints/latent_dim_{latent_dim}')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')

    print('Start Generating')
    # Clean up the 'samples' directory if it exists
    if os.path.exists('EM_GM_samples'):
        shutil.rmtree('EM_GM_samples')
    os.makedirs('EM_GM_samples', exist_ok=True)

    # Load Laplace Mixture Models and Scaler
    with open('laplace_mixtures.pkl', 'rb') as f:
        laplace_mixtures = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    total_samples = 10000
    n_samples_generated = 0

    with torch.no_grad():
        while n_samples_generated < total_samples:
            current_batch_size = min(args.batch_size, total_samples - n_samples_generated)
            
            # Sample from each dimension's Laplace Mixture Model
            z_samples_std = np.zeros((current_batch_size, latent_dim))
            for dim in range(latent_dim):
                lm = laplace_mixtures[dim]
                z_samples_std[:, dim] = lm.sample(current_batch_size)

            # Inverse transform the standardized samples
            z_samples = scaler.inverse_transform(z_samples_std)

            # Convert to PyTorch Tensor and Move to GPU
            z = torch.from_numpy(z_samples).float().cuda()

            # Generate Images
            x = model(z)
            x = x.view(current_batch_size, 1, 28, 28)  # Adjust if your images have different dimensions
            
            # Save Images
            for i in range(current_batch_size):
                torchvision.utils.save_image(
                    x[i], 
                    os.path.join('EM_GM_samples', f'{n_samples_generated + i}.png')
                )
            
            n_samples_generated += current_batch_size
            print(f'Generated {n_samples_generated}/{total_samples} images')

    print("Image generation completed.")