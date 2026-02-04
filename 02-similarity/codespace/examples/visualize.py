import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load Plato embeddings
with open('data/Plato/TheRepublic_embeddings.json', 'r', encoding='utf-8') as f:
    plato_data = json.load(f)

# Load Van Gogh embeddings
with open('data/VanGogh/VanGogh_embeddings.json', 'r', encoding='utf-8') as f:
    vangogh_data = json.load(f)

# Load Picasso embeddings
with open('data/Picasso/Picasso_embeddings.json', 'r', encoding='utf-8') as f:
    picasso_data = json.load(f)

# Extract embeddings
plato_embeddings = np.array([item['embedding'] for item in plato_data])
vangogh_embeddings = np.array([item['embedding'] for item in vangogh_data])
picasso_embeddings = np.array([item['embedding'] for item in picasso_data])

# Combine all embeddings for PCA
all_embeddings = np.vstack([plato_embeddings, vangogh_embeddings, picasso_embeddings])

print(f"Loaded {len(plato_embeddings)} Plato embeddings")
print(f"Loaded {len(vangogh_embeddings)} Van Gogh embeddings")
print(f"Loaded {len(picasso_embeddings)} Picasso embeddings")
print(f"Total embeddings: {len(all_embeddings)} with dimension {all_embeddings.shape[1]}")

# Apply PCA for dimensionality reduction to 2D on all data at once
print("Applying PCA dimensionality reduction...")
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(all_embeddings)

# Split the 2D embeddings back into Plato, Van Gogh, and Picasso
plato_2d = embeddings_2d[:len(plato_embeddings)]
vangogh_2d = embeddings_2d[len(plato_embeddings):len(plato_embeddings) + len(vangogh_embeddings)]
picasso_2d = embeddings_2d[len(plato_embeddings) + len(vangogh_embeddings):]

# Create the plot
plt.figure(figsize=(12, 8))

# Plot Plato as circles
plt.scatter(
    plato_2d[:, 0],
    plato_2d[:, 1],
    marker='o',
    s=5,
    c='#4169E1',
    label='Plato (The Republic)',
    edgecolors='none',
    alpha=0.6
)

# Plot Van Gogh as stars
plt.scatter(
    vangogh_2d[:, 0],
    vangogh_2d[:, 1],
    marker='*',
    s=200,
    c='#FFD700',
    label='Van Gogh',
    edgecolors='black',
    linewidths=0.5
)

# Plot Picasso as squares
plt.scatter(
    picasso_2d[:, 0],
    picasso_2d[:, 1],
    marker='s',
    s=100,
    c='#FF69B4',
    label='Picasso',
    edgecolors='black',
    linewidths=0.5
)

plt.title('PCA Visualization of Plato, Van Gogh, and Picasso Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(loc='upper right')
plt.tight_layout()

# Save the plot
output_file = 'combined_embeddings_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_file}")

plt.show()
