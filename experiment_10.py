##Clustering by applying t-SNE to the GW-barycentric coordinate space:

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode

import ot

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()


## GETTING RANDOM TEMPLATES FROM DATASET ##########################################################
# Templates are of the form (matrix, measure)
n_classes = 10 #Since we are working with digits 0-9
n_temp = 2  # Number of templates for each digit
ind_temp_list = []  # list of template indices from dataset
measure_temp_list = []  # list of template measures
matrix_temp_list = []  # list of template dissimilarity matrices

for digit in range(n_classes):
    for s in range(n_temp):
        # Select a random index corresponding to the chosen digit
        ind = digit_indices[digit][np.random.randint(len(digit_indices[digit]))]
        ind_temp_list.append(ind)

        # Extract the probability measure from the third column of Data (p_s)
        p_s = Data[ind, :, 2]

        # Find valid indices where p_s is not -1 (avoiding missing or padded values)
        valid_indices = np.where(p_s != -1)[0]

        # Filter out invalid entries from p_s
        p_s = p_s[valid_indices]

        # Normalize p_s to make it a valid probability distribution
        p_s = p_s / float(p_s.sum())
        measure_temp_list.append(p_s)

        # Extract spatial coordinates (first two columns of Data) for valid points
        C_s = Data[ind, valid_indices, :2]

        # Center the points by subtracting the mean
        # and Normalize coordinates to fit within the unit square [0,1]²
        C_s = utils.normalize_2Dpointcloud_coordinates(C_s)

        # Compute the pairwise Euclidean distance matrix for C_s
        dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)
        matrix_temp_list.append(dist_matrix_s)

print('Random templates, extracted')

## PLOT TEMPLATES #################################################################################
fig, axes = plt.subplots(1, 10*n_temp, figsize=(15, 10))
axes = axes.flatten()

for i, ind in enumerate(ind_temp_list):
    a = Data[ind, :, 2]
    X = Data[ind, a != -1, :2]
    X = utils.normalize_2Dpointcloud_coordinates(X)
    a = a[a != -1]
    a = a / float(a.sum())
    axes[i].scatter(X[:, 0], X[:, 1], s=a * 250)
    #axes[i].set_title(f'Template #{i + 1}')
    axes[i].set_aspect('equal', adjustable='box')
    axes[i].set_xticks([])  # Remove x-axis ticks
    axes[i].set_yticks([])  # Remove y-axis ticks
# Add figure title
fig.suptitle("Templates", fontsize=16)
plt.show()




## GET TRAINING SAMPLES FROM DATASET ##############################################################
# Convert to NumPy arrays
Data = np.array(Data)  # Shape: (num_samples, num_points, 3)
label = np.array(label)  # Shape: (num_samples,)

# Split into training and test sets (test_size% test, (100-test_size)% training)
X_train, X_test, y_train, y_test = train_test_split(Data, label, test_size=0.995, random_state=42, stratify=label)

# Initialize lists for training set
train_indices = []
train_measures = []
train_distance_matrices = []

# Process the training set
for i in range(X_train.shape[0]):
    train_indices.append(i)

    # Extract probability measure (third column) and filter valid points
    p_s = X_train[i, :, 2]
    valid_indices = np.where(p_s != -1)[0]
    p_s = p_s[valid_indices]

    # Normalize p_s to make it a valid probability distribution
    p_s = p_s / float(p_s.sum())
    train_measures.append(p_s)

    # Extract and normalize spatial coordinates (first two columns)
    C_s = X_train[i, valid_indices, :2]
    C_s = utils.normalize_2Dpointcloud_coordinates(C_s)

    # Compute pairwise Euclidean distance matrix
    dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)
    train_distance_matrices.append(dist_matrix_s)

print('Train samples, extracted')
print(f"Number of training sample points: {len(train_distance_matrices)}")




## APPLY THE GW-BARYCENTER ANALYSIS METHOD: EXTRACT GW-BARYCENTER COORDINATES (lambdas) ###########
print('Compute GW-barycentric coordinates')

# Initialize the list to store rows of the big matrix
lambda_matrix_list = []

# Compute lambdas for each (B, b) pair in training data
for i in range(len(train_distance_matrices)):
    B = train_distance_matrices[i]  # Distance matrix
    b = train_measures[i]  # Measure
    label = y_train[i]  # Label of the training point

    # Compute lambdas using the given function
    _, lambdas = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, b)

    # Store the result in a row (label first, then lambda values)
    lambda_matrix_list.append(np.concatenate(([label], lambdas)))

    # Print progress every 50 iterations
    if i % 50 == 0:
        print(f"Processed {i} samples...")

# Convert the list to a NumPy array (final big matrix)
lambda_matrix = np.array(lambda_matrix_list)

# Print shape of the resulting matrix
print('Barycentric coordinates, computed')


labels = lambda_matrix[:,0:1].squeeze() ## First Column has the labels of the training set
matrix = lambda_matrix[:,1:] ## The rest of the columns correspond to GW-barycentric coordinates. Each row corresponds to one training sample from the data set.


## APPLY t-SNE ####################################################################################
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedded = tsne.fit_transform(matrix)
print('t-SNE in the barycenter coordinates, done')

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
num_clusters = 10  # Set the number of clusters (digits 0-9)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
predicted_labels = kmeans.fit_predict(embedded)  # Cluster assignments

# True labels from the training set
true_labels = labels

# Map clusters to true labels
mapped_labels = np.zeros_like(labels)

for cluster in range(num_clusters):
    mask = (predicted_labels == cluster)
    if np.any(mask):  # Avoid empty clusters
        mapped_labels[mask] = mode(true_labels[mask])[0]  # Most common true label in the cluster

# Compute accuracy
accuracy = accuracy_score(true_labels, mapped_labels)
print(f"Clustering Accuracy: {accuracy:.4f}")



