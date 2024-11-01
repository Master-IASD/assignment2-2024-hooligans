import os
import shutil
import pandas as pd

# Set paths and variables
working_dir = "D:\\iasd\\dslab\\p2\\assignment2-2024-hooligans"
samples_folder = "GM_samples"
table_labels = "tables\\GM_samples_with_resnet.csv"
class_samples_dir = os.path.join(working_dir, "class_samples")
sample_subdir = os.path.join(class_samples_dir, samples_folder)

# Create class_samples and subdirectories if not existing
os.makedirs(sample_subdir, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(sample_subdir, str(i)), exist_ok=True)

# Load label table
labels_df = pd.read_csv(os.path.join(working_dir, table_labels))

# Copy images based on labels
for _, row in labels_df.iterrows():
    image_file = row['image'] # or 'Image'
    label = str(row['label']) # or 'Label'
    src_path = os.path.join(working_dir, samples_folder, image_file)
    dst_path = os.path.join(sample_subdir, label, image_file)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)