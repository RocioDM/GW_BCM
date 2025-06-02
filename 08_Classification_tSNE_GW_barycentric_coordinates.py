## Clustering by applying t-SNE to the GW-barycentric coordinate space
## This notebook uses the 2D point cloud MNIST dataset

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
import seaborn as sns
from matplotlib.colors import ListedColormap


## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()


# Select some digits
selected_digits = [0,4]
selected_indices = np.concatenate([digit_indices[d] for d in selected_digits])

# Filter the dataset
Data_selected = Data[selected_indices]
label_selected = label[selected_indices]


## GETTING RANDOM TEMPLATES FROM DATASET ##########################################################
# Templates are of the form (matrix, measure)
n_classes = len(selected_digits)
n_temp = 1   # Number of templates for each digit
ind_temp_list = []  # list of template indices from dataset
measure_temp_list = []  # list of template measures
matrix_temp_list = []  # list of template dissimilarity matrices

for digit in selected_digits:
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
fig, axes = plt.subplots(1, n_classes*n_temp, figsize=(8, 4))
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
plt.tight_layout()
plt.savefig("templates.pdf", bbox_inches='tight')
plt.show()




## GET TRAINING SAMPLES FROM DATASET ##############################################################
# Split into training and test sets (test_size% test, (100-test_size)% training)
X_train, X_test, y_train, y_test = train_test_split(Data_selected, label_selected, test_size=0.05, random_state=42, stratify=label_selected)


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

# Ask the user to choose a method
method = int(input("Choose a method: fixed-point approach [enter 1] or gradient approach via blow-up [enter 2] "))


# Compute lambdas for each (B, b) pair in training data
for i in range(len(train_distance_matrices)):
    B = train_distance_matrices[i]  # Distance matrix
    b = train_measures[i]  # Measure
    label = y_train[i]  # Label of the training point

    # Compute lambdas using the given function
    if method == 1:
        _, lambdas = utils.get_lambdas(matrix_temp_list, measure_temp_list, B, b)   #Fixed-point approach

    elif method == 2:
        B, b, temp_blow_up = utils.blow_up(matrix_temp_list, measure_temp_list, B, b)   #Templates blow-up
        _, lambdas = utils.get_lambdas_blowup(temp_blow_up, B, b) #Gradient approach


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
tsne = TSNE(n_components=2, perplexity=n_classes*n_temp, random_state=42)
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
plt.xticks([])  # Remove x ticks
plt.yticks([])  # Remove y ticks
plt.tight_layout()
plt.savefig("tsne.pdf", bbox_inches='tight')
plt.show()


# Apply K-Means clustering
num_clusters = n_classes  # Set the number of clusters (digits 0-9)
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
print(f"Overall Clustering Accuracy: {accuracy:.4f}")



# Scatter plot of clusters
colors = plt.colormaps.get_cmap("tab10").colors[:n_classes]  # Extract 'n_classes' colors from 'tab10'
custom_cmap = ListedColormap(colors)  # Create a colormap with only 'n_classes' colors

scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=predicted_labels, cmap=custom_cmap, alpha=0.7)
plt.colorbar(scatter, ticks=range(n_classes), label="Cluster ID")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.title("K-Means Clustering Visualization")
plt.tight_layout()
plt.savefig("KMeans.pdf", bbox_inches='tight')
plt.show()


# Compute confusion matrix (optional)
conf_matrix = confusion_matrix(true_labels, mapped_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(n_classes), yticklabels=range(n_classes))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for K-Means Clustering")
plt.tight_layout()
plt.savefig("confusion_matrix.pdf", bbox_inches='tight')
plt.show()



# Compute per-class accuracy
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Print per-class accuracy
for class_idx, acc in enumerate(class_accuracies):
    print(f"Accuracy for Class {class_idx}: {acc:.4f}")


