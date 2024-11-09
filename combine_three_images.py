from PIL import Image, ImageDraw, ImageFont

# Paths to your images
image_paths = ["figures/Fig_3_generated_mnist_tsne_vanillaGAN_dim_200.png", "figures/Fig_4_real_mnist_tsne.png", "figures/Fig_7_GM_samples_tsne.png"]
titles = ["Baseline", "Real MNIST", "Best GM GAN"]

# Load images
images = [Image.open(img_path) for img_path in image_paths]
widths, heights = zip(*(i.size for i in images))

# Calculate total width and max height for new image
total_width = sum(widths)
max_height = max(heights)

# Define title height and font (adjust path to your font if necessary)
title_height = 50
font = ImageFont.truetype("arial.ttf", 24)  # Use a path to a .ttf font file if needed

# Create new image with extra space for titles
combined_image = Image.new("RGBA", (total_width, max_height + title_height), (255, 255, 255, 255))
draw = ImageDraw.Draw(combined_image)

# Position images and titles
x_offset = 0
for i, img in enumerate(images):
    combined_image.paste(img, (x_offset, title_height))  # Paste image below title space
    title_text = titles[i]
    text_bbox = draw.textbbox((0, 0), title_text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_x = x_offset + (img.width - text_width) // 2
    draw.text((text_x, (title_height - text_height) // 2), title_text, font=font, fill="black")
    x_offset += img.width

# Save the combined image
combined_image.save("combined_image.png")

