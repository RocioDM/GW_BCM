### Generic t-SNE


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

## IMPORT USER DEFINED LIBRARIES ##################################################################
from tensorflow.keras.datasets import mnist

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Combine training and test images
X = np.vstack((X_train, X_test))
y = np.hstack((y_train, y_test))

# Flatten each image into a row of 784 pixels (28x28)
X_flattened = X.reshape(X.shape[0], -1)  # Shape: (70000, 784)

# Combine the label and image pixels into a matrix
# First column: label, Remaining 784 columns: image pixels
mnist_matrix = np.hstack((y.reshape(-1, 1), X_flattened))

# Save as CSV (optional)
#np.savetxt("mnist_matrix.csv", mnist_matrix, delimiter=",", fmt='%d')

# Display shape
print("Matrix shape:", mnist_matrix.shape)  # (70000, 785)

labels = mnist_matrix[:10000,0:1].squeeze()

print(labels.shape)
matrix = mnist_matrix[:10000,1:]


# Perform PCA
n_components = 30  # Reduce dimensions
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(matrix)
print('PCA done')
# Explained variance ratio
# explained_variance = pca.explained_variance_ratio_
# print(f"Explained Variance Ratio: {explained_variance}")

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedded = tsne.fit_transform(X_pca)

# Plot the result
plt.figure(figsize=(10, 7))
for label in np.unique(labels):
    plt.scatter(embedded[labels == label, 0], embedded[labels == label, 1], label=f'Class {label}', alpha=0.7, edgecolors='k')
plt.title('t-SNE Projection into 2D')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()

plt.show()



# Apply K-Means clustering
num_clusters = 10  # Set the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embedded)


# Plot clustered data
plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='viridis', edgecolors='k', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, color='red', label='Centers')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("K-Means Clustering Visualization")
plt.legend()
plt.show()