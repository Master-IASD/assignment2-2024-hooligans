import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

# Move the ImageDataset class outside the main block
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("L")  # Convert to grayscale
            image = self.transform(image)
            return image, image_path
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None  # Return None if there's an error

# Move the collate_fn function outside the main block
def collate_fn(batch):
    # Filter out None samples
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return torch.empty(0), []
    images, paths = zip(*batch)
    images = torch.stack(images, 0)
    return images, paths

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transformation for MNIST images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),                # Resize to ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081))  # MNIST normalization
    ])

    # Build a DataFrame with 10,000 PNG file names
    #samples_folder = "samples"
    samples_folder = "GM_samples"
    images = [f"{samples_folder}/{i}.png" for i in range(10000)]
    df = pd.DataFrame()
    df["image"] = images

    # Create the dataset and dataloader
    dataset = ImageDataset(df['image'].tolist(), transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    # Load the pre-trained ResNet model from resnet_checkpoints
    checkpoint_dir = 'resnet_checkpoints'
    # Get the list of checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    checkpoint_files.sort()  # Ensure the checkpoints are sorted
    latest_checkpoint = checkpoint_files[-1]  # Use the latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    # Initialize the model architecture using the updated 'weights' parameter
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # MNIST has 10 classes
    model = model.to(device)

    # Load the model weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set the model to evaluation mode

    # Process images in batches
    labels = []
    image_paths = []

    for images_batch, paths_batch in dataloader:
        if images_batch.size(0) == 0:
            continue  # Skip empty batches
        images_batch = images_batch.to(device)

        with torch.no_grad():
            outputs = model(images_batch)
            _, predicted = torch.max(outputs, 1)

        labels.extend(predicted.cpu().tolist())
        image_paths.extend(paths_batch)

    # Create DataFrame with valid image paths and labels
    df = pd.DataFrame({'image': image_paths, 'label': labels})

    # Plot the distribution of classes
    class_counts = df['label'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar')
    plt.xlabel('Class Label')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Classes in the 10,000 Images')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{samples_folder}_class_distribution.png')
    plt.show()

    # --- New code to plot 100 images with their predicted labels ---

    # Sample 100 images from the DataFrame
    sampled_df = df.sample(n=100, random_state=42).reset_index(drop=True)

    # Create a grid for plotting
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    axes = axes.flatten()

    for idx, (img_path, label) in enumerate(zip(sampled_df['image'], sampled_df['label'])):
        image = Image.open(img_path).convert("L")
        axes[idx].imshow(image, cmap='gray')
        axes[idx].set_title(f"Label: {label}")
        axes[idx].axis('off')

    plt.tight_layout()
    # Save the figure
    plt.savefig(f'{samples_folder}_with_labels.png')
    plt.show()

    # --- New code to save the data frame as csv ---

    df.to_csv(f'tables/{samples_folder}_with_resnet.csv', index=False)