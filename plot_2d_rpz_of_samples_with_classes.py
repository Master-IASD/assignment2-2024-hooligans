import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load your DataFrame
#df = pd.read_csv("tables/vanilla_gan_dim_200_with_resnet.csv")
df = pd.read_csv("tables/GM_samples_with_resnet.csv")

# Read and flatten the images
X = []
for img_path in df['image']:
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img_arr = np.array(img).flatten()
    X.append(img_arr)
X = np.array(X)

# Extract labels
y = df['label'].values

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot the t-SNE result
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=5)
plt.legend(handles=scatter.legend_elements()[0], labels=range(10), title="Classes")
plt.title('t-SNE of MNIST Images')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.colorbar()
plt.savefig('figures/mnist_tsne.png')
plt.show()