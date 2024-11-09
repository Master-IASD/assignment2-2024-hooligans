import os
import random
from PIL import Image

# Set paths
real_samples_folder = "real_samples"
regenerated_folder = "regenerated"
output_path = "sample_comparison.png"

# Select 20 random indices
indices = random.sample(range(10000), 20)

# Load image pairs and concatenate vertically
images = []
for i in indices:
    real_image_path = os.path.join(real_samples_folder, f"image_{i}.png")
    regenerated_image_path = os.path.join(regenerated_folder, f"image_{i}_regenerated.png")

    # Open and concatenate images top-bottom
    real_image = Image.open(real_image_path)
    regenerated_image = Image.open(regenerated_image_path)
    combined_image = Image.new('RGB', (real_image.width, real_image.height + regenerated_image.height))
    combined_image.paste(real_image, (0, 0))
    combined_image.paste(regenerated_image, (0, real_image.height))
    images.append(combined_image)

# Stack all pairs horizontally
total_width = sum(img.width for img in images)
max_height = max(img.height for img in images)
final_image = Image.new('RGB', (total_width, max_height))

# Paste each pair into the final image
x_offset = 0
for img in images:
    final_image.paste(img, (x_offset, 0))
    x_offset += img.width

# Save the output
final_image.save(output_path)