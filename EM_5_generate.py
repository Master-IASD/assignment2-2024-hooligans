import torch 
import torchvision
import os
import shutil
import argparse
import joblib

from model import Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Images using GMM.')
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="The batch size to use for generation.")
    args = parser.parse_args()
    latent_dim = 150
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

    # **Load GMM and Scaler**
    gmm = joblib.load('gmm_model.pkl')
    scaler = joblib.load('gm_scaler.pkl')  # Ensure you saved the scaler during training

    total_samples = 10000
    n_samples_generated = 0

    with torch.no_grad():
        while n_samples_generated < total_samples:
            current_batch_size = min(args.batch_size, total_samples - n_samples_generated)
            
            # **Correct Sampling from GMM**
            z_samples, _ = gmm.sample(current_batch_size)  # Sample from the GMM
            z_samples_std, _ = gmm.sample(current_batch_size)  # Sample from the GMM
            z_samples = scaler.inverse_transform(z_samples_std)  # Inverse the standardization
            
            # **Convert to PyTorch Tensor and Move to GPU**
            z = torch.from_numpy(z_samples).float().cuda()
            #z = torch.randn(current_batch_size, latent_dim).cuda()


            # **Generate Images**
            x = model(z)
            x = x.view(current_batch_size, 1, 28, 28)  # Reshape to image dimensions
            
            # **Save Images**
            for i in range(current_batch_size):
                torchvision.utils.save_image(
                    x[i], 
                    os.path.join('EM_GM_samples', f'{n_samples_generated + i}.png')
                )
            
            n_samples_generated += current_batch_size
            print(f'Generated {n_samples_generated}/{total_samples} images')
