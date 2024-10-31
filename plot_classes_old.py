import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from torchvision.datasets import MNIST

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build a data frame with 10,000 PNG file names
    images = [f"samples/{i}.png" for i in range(10000)]
    df = pd.DataFrame()
    df["image"] = images

    # Define the transformation for MNIST-like images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB for ResNet
        transforms.Resize((224, 224)),                # Resize to ResNet's input size
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))  # MNIST normalization
    ])

    # Load a pre-trained ResNet model and modify it for 10 classes
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    nn.init.xavier_uniform_(model.fc.weight)
    model = model.to(device)

    # Load MNIST dataset
    train_dataset = MNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = MNIST(root='data', train=False, transform=transform, download=True)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Fine-tune the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 5  # Define the number of epochs

    # Training loop
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images_batch, labels_batch in train_loader:
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
            # Forward pass
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Print loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    # Set the model to evaluation mode after training
    model.eval()

    # Save the trained model
    torch.save(model.state_dict(), 'mnist_resnet18.pth')

    # Function to predict the label of an image
    def get_image_label(image_path, model):
        image = Image.open(image_path).convert("L")  # Open and convert to grayscale
        image = transform(image).unsqueeze(0).to(device)  # Apply transformations and move to device
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        return predicted.item()

    # Generate labels for all images in the DataFrame
    df["label"] = [get_image_label(img, model) for img in df["image"]]

    # Sample 100 images and labels from the DataFrame
    sampled_df = df.sample(100)

    # Plot images with labels
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    axes = axes.flatten()

    for i, (img_path, label) in enumerate(zip(sampled_df["image"], sampled_df["label"])):
        image = mpimg.imread(img_path)
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()