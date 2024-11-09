from PIL import Image, ImageDraw, ImageFont

# Load images
image1 = Image.open("figures/Fig_2_class_distribution_vanilla_dim_200.png")
image2 = Image.open("figures/Fig_6_GM_samples_class_distribution.png")

# Titles for each sub-image
title1 = "Baseline (Vanilla GAN with d=200)"
title2 = "Dynamic Supervised GM GAN"

# Font settings (modify path if needed)
try:
    font = ImageFont.truetype("arial.ttf", 20)  # Common font, may need adjustment if not found
except IOError:
    font = ImageFont.load_default()

# Calculate title heights
title_height = max(font.getbbox(title1)[3], font.getbbox(title2)[3])

# Determine dimensions for new image with titles and separation line
total_width = image1.width + image2.width
max_image_height = max(image1.height, image2.height)
line_height = 5  # Thickness of the separation line
total_height = title_height + max_image_height + line_height

# Create new blank image
new_image = Image.new("RGBA", (total_width, total_height), (255, 255, 255, 255))
draw = ImageDraw.Draw(new_image)

# Draw titles
draw.text((image1.width // 2 - font.getbbox(title1)[2] // 2, 0), title1, font=font, fill="black")
draw.text((image1.width + image2.width // 2 - font.getbbox(title2)[2] // 2, 0), title2, font=font, fill="black")

# Paste images below titles
new_image.paste(image1, (0, title_height))
new_image.paste(image2, (image1.width, title_height))

# # Draw separation line
# draw.line([(image1.width, title_height), (image1.width, total_height)], fill="black", width=line_height)

# Save the combined image
new_image.save("combined_image_with_titles.png")
