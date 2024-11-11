import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import struct
import os

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

if __name__ == "__main__":
    # Define the paths to the MNIST files
    data_dir = r"D:\iasd\dslab\p2\assignment2-2024-hooligans\data\MNIST\raw"
    test_images_file = os.path.join(data_dir, "t10k-images-idx3-ubyte")
    test_labels_file = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

    # Load test images and labels
    x_test = load_mnist_images(test_images_file)
    y_test = load_mnist_labels(test_labels_file)

    # Normalize the pixel values to [0, 1]
    X = x_test.astype('float32') / 255.0

    # Apply t-SNE to reduce dimensions to 2
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plot the t-SNE result
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test, cmap='tab10', s=5)
    plt.colorbar()
    plt.legend(handles=scatter.legend_elements()[0], labels=range(10), title="Classes")
    plt.title('t-SNE of 10,000 MNIST Images')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('figures/real_mnist_tsne.png')
    plt.show()