import os
import torch
import torch.optim as optim 
import torchvision
from PIL import Image
import multiprocessing
from functools import partial

from model import Generator
from utils import load_model

def process_image(image_file, model, latent_dim, real_samples_folder, z_real_gen_folder, regenerated_folder):
    print(f'Processing {image_file}...')
    
    # Load and preprocess the image
    image_path = os.path.join(real_samples_folder, image_file)
    image = Image.open(image_path).convert('L')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
    ])
    x = transform(image).view(1, -1).cuda()
    x = x * 2 - 1

    # Initialize latent vector z
    batch_size = 1
    z_0 = torch.randn(batch_size, latent_dim).cuda()
    z = z_0.clone().detach().requires_grad_(True)
    
    # Define custom loss function
    def custom_loss(z, x, model):
        return torch.norm(x - model(z), p=2)
    
    # Set up optimizer
    optimizer = optim.Adam([z], lr=0.01)
    
    # Optimization loop
    num_steps = 1000
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = custom_loss(z, x, model)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0 or step == num_steps - 1:
            print(f'{image_file} - Step {step}, Loss: {loss.item()}')

    print(f'Optimization finished for {image_file}.')

    # Save the optimized z tensor
    z_filename = os.path.splitext(image_file)[0] + '_z.pt'
    z_path = os.path.join(z_real_gen_folder, z_filename)
    torch.save(z.detach().cpu(), z_path)
    print(f'Saved optimized z to {z_path}')

    # Optional: Generate the image from the optimized z and save it
    x_gen = model(z)
    x_gen = x_gen.reshape(batch_size, 28, 28)

    # Save the generated image as a PNG file
    regenerated_image_filename = os.path.splitext(image_file)[0] + '_regenerated.png'
    regenerated_image_path = os.path.join(regenerated_folder, regenerated_image_filename)
    torchvision.utils.save_image(x_gen, regenerated_image_path)

    print(f'Regenerated image saved to {regenerated_image_path}')

    # Optionally, print min and max values
    print(f"{image_file} - x min: {x.min().item()}, x max: {x.max().item()}")
    print(f"{image_file} - model(z) min: {model(z).min().item()}, model(z) max: {model(z).max().item()}")

def main():
    latent_dim = 150

    # Model Pipeline
    mnist_dim = 784
    model = Generator(g_output_dim=mnist_dim, latent_dim=latent_dim).cuda()
    model = load_model(model, f'checkpoints/latent_dim_{latent_dim}')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')

    # Create the 'z_real_gen' and 'regenerated' folders if they don't exist
    z_real_gen_folder = 'z_real_gen'
    regenerated_folder = 'regenerated'
    os.makedirs(z_real_gen_folder, exist_ok=True)
    os.makedirs(regenerated_folder, exist_ok=True)

    # Get the list of image files in 'real_samples' folder
    real_samples_folder = 'real_samples'
    image_files = [f for f in os.listdir(real_samples_folder) if f.endswith('.png') or f.endswith('.jpg')]

    # Prepare partial function for multiprocessing
    partial_process_image = partial(
        process_image,
        model=model,
        latent_dim=latent_dim,
        real_samples_folder=real_samples_folder,
        z_real_gen_folder=z_real_gen_folder,
        regenerated_folder=regenerated_folder
    )

    # Use multiprocessing Pool to parallelize processing
    num_processes = min(multiprocessing.cpu_count(), len(image_files))
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(partial_process_image, image_files)

    print('All images processed.')

if __name__ == '__main__':
    main()
