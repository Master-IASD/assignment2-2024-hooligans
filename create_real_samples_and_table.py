import os
import struct
import numpy as np
import pandas as pd
from PIL import Image

if __name__ == "__main__":

    # Paths
    base_dir = "D:\\iasd\\dslab\\p2\\assignment2-2024-hooligans"
    data_dir = os.path.join(base_dir, "data", "MNIST", "MNIST", "raw")
    output_dir = os.path.join(base_dir, "real_samples")
    table_dir = os.path.join(base_dir, "tables")

    # Create directories if not exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # File paths
    image_file_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
    label_file_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")
    table_file_path = os.path.join(table_dir, "image_labels.csv")

    # Function to read IDX format images and labels
    def read_idx_images(file_path):
        with open(file_path, "rb") as f:
            _, _, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols)
        return images

    def read_idx_labels(file_path):
        with open(file_path, "rb") as f:
            f.read(8)
            labels = np.fromfile(f, dtype=np.uint8)
        return labels

    # Load images and labels
    images = read_idx_images(image_file_path)
    labels = read_idx_labels(label_file_path)

    # Save images as PNG and record names and labels
    records = []
    for i, (image, label) in enumerate(zip(images, labels)):
        image_name = f"image_{i}.png"
        image_path = os.path.join(output_dir, image_name)
        Image.fromarray(image).save(image_path)
        records.append({"Image": image_name, "Label": label})

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(table_file_path, index=False)