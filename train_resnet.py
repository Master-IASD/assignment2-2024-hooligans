import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import MNIST
import os

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transformation for MNIST images
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),                # Resize to ResNet input size
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
    num_epochs = 5  # Adjust the number of epochs as needed

    # Create the directory to save checkpoints if it doesn't exist
    checkpoint_dir = 'resnet_checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

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
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save the model checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'mnist_resnet18_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

    print("Training complete. Model checkpoints are saved in the 'resnet_checkpoints' directory.")