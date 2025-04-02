##Clustering by applying t-SNE to the GW-barycentric coordinate space:

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
import seaborn as sns
from matplotlib.colors import ListedColormap



import ot

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()


# Select only some digits
selected_digits = [0, 1]
selected_indices = np.concatenate([digit_indices[d] for d in selected_digits])

# Filter the dataset
Data_selected = Data[selected_indices]
label_selected = label[selected_indices]


## GETTING RANDOM TEMPLATES FROM DATASET ##########################################################
# Templates are of the form (matrix, measure)
n_classes = len(selected_digits)
n_temp = 1  # Number of templates for each digit
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
fig, axes = plt.subplots(1, n_classes*n_temp, figsize=(15, 10))
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
# # Convert to NumPy arrays
# Data = np.array(Data)  # Shape: (num_samples, num_points, 3)
# label = np.array(label)  # Shape: (num_samples,)

# Split into training and test sets (test_size% test, (100-test_size)% training)
#X_train, X_test, y_train, y_test = train_test_split(Data, label, test_size=0.995, random_state=42, stratify=label)
X_train, X_test, y_train, y_test = train_test_split(Data_selected, label_selected, test_size=0.52, random_state=42, stratify=label_selected)


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

    if lambdas.min()>=0.0:
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


# Extract first two dimensions of matrix as coordinates
X = matrix[:, 0]  # First dimension (x-axis)
Y = matrix[:, 1]  # Second dimension (y-axis)

# Define colors for labels (0: blue, 1: red)
custom_cmap = ListedColormap(["blue", "red"])

# Create scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X, Y, c=labels, cmap=custom_cmap, alpha=0.7, edgecolors="k")

# Add legend manually
legend_labels = {0: "Class 0", 1: "Class 1"}
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=legend_labels[l])
           for l, color in zip([0, 1], ["blue", "red"])]
plt.legend(handles=handles)

# Axis labels and title
plt.xlabel("Barycentric Coordinate 1")
plt.ylabel("Barycentric Coordinate 2")
plt.title("Barycentric Coordinates Scatter Plot")

# Show plot
plt.show()