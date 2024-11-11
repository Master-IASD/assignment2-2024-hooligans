import os
import torch
import torch.optim as optim 
import torchvision
from PIL import Image

from model import Generator
from utils import load_model

if __name__ == '__main__':
    latent_dim = 150
    batch_size = 1

    # Model Pipeline
    mnist_dim = 784
    model = Generator(g_output_dim=mnist_dim, latent_dim=latent_dim).cuda()
    model = load_model(model, f'checkpoints/latent_dim_{latent_dim}')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')

    # Create the 'z_real_gen' folder if it doesn't exist
    os.makedirs('z_real_gen', exist_ok=True)

    # Get the list of image files in 'real_samples' folder
    real_samples_folder = 'real_samples'
    image_files = [f for f in os.listdir(real_samples_folder) if f.endswith('.png') or f.endswith('.jpg')]

    # Loop over each image
    for image_file in image_files:
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
                print(f'Step {step}, Loss: {loss.item()}')

        print('Optimization finished.')

        # Save the optimized z tensor
        z_filename = os.path.splitext(image_file)[0] + '_z.pt'
        z_path = os.path.join('z_real_gen', z_filename)
        torch.save(z.detach().cpu(), z_path)
        print(f'Saved optimized z to {z_path}')

        # Optional: Generate the image from the optimized z and save it
        x_gen = model(z)
        x_gen = x_gen.reshape(batch_size, 28, 28)

        # Create the 'regenerated' folder if it doesn't exist
        os.makedirs('regenerated', exist_ok=True)

        # Save the generated image as a PNG file
        regenerated_image_filename = os.path.splitext(image_file)[0] + '_regenerated.png'
        regenerated_image_path = os.path.join('regenerated', regenerated_image_filename)
        torchvision.utils.save_image(x_gen, regenerated_image_path)

        print(f'Regenerated image saved to {regenerated_image_path}')

        # Optionally, print min and max values
        print("x min:", x.min().item(), "x max:", x.max().item())
        print("model(z) min:", model(z).min().item(), "model(z) max:", model(z).max().item())

    print('All images processed.')
