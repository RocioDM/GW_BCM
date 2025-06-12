## GW-barycentric coordinate space with 3 classes
## In this notebook uses the 2D Point Cloud MNIST dataset

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()


# Select only some digits
selected_digits = [0, 4,7]

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
fig, axes = plt.subplots(1, n_classes*n_temp, figsize=(8, 5))
axes = axes.flatten()

for i, ind in enumerate(ind_temp_list):
    a = Data[ind, :, 2]
    X = Data[ind, a != -1, :2]
    X = utils.normalize_2Dpointcloud_coordinates(X)
    a = a[a != -1]
    a = a / float(a.sum())
    axes[i].scatter(X[:, 0], X[:, 1], s=a * 350)
    #axes[i].set_title(f'Template #{i + 1}')
    axes[i].set_aspect('equal', adjustable='box')
    axes[i].set_xticks([])  # Remove x-axis ticks
    axes[i].set_yticks([])  # Remove y-axis ticks
    coord = ((1 - i) * (2 - i) * 0.5 * np.array([-1, 0]) +
             (2 - i) * i * np.array([1, 0]) +
             (i - 1) * i * 0.5 * np.array([0,3 ** 0.5]))
    axes[i].set_title(f'Template Label = {int(label[ind])}\n Coord {coord}')
# Add figure title
fig.suptitle("Templates", fontsize=16)
plt.show()




## GET TRAINING SAMPLES FROM DATASET ##############################################################
# # Convert to NumPy arrays
# Data = np.array(Data)  # Shape: (num_samples, num_points, 3)
# label = np.array(label)  # Shape: (num_samples,)

# Split into training and test sets (test_size% test, (100-test_size)% training)
X_train, X_test, y_train, y_test = train_test_split(Data_selected, label_selected, test_size=0.9, random_state=42, stratify=label_selected)


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

    # Compute lambdas
    _, lambdas = utils.get_lambdas_constraints(matrix_temp_list, measure_temp_list, B, b)

    if lambdas.min()>=0.0:
        # Store the result in a row (label first, then lambda values)
        lambda_matrix_list.append(np.concatenate(([label], lambdas)))

    # Print progress every 50 iterations
    if i % 50 == 0:
        print(f"Processed {i} samples...")

# Convert the list to a NumPy array (final big matrix)
lambda_matrix = np.array(lambda_matrix_list)
print('Barycentric coordinates, computed')

# Predict labels using argmax of barycentric coordinates
labels = lambda_matrix[:,0:1].squeeze() ## First Column has the labels of the training set
matrix = lambda_matrix[:,1:] ## The rest of the columns correspond to GW-barycentric coordinates. Each row corresponds to one training sample from the data set.

predicted_labels = np.argmax(matrix, axis=1)
accuracy = np.mean(predicted_labels == labels)
print(f"Classification Accuracy on Training Set (using argmax of GW-barycentric coordinates): {accuracy:.4f}")



cm = confusion_matrix(labels, predicted_labels, labels=[0,1,2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Digit {d}' for d in selected_digits])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Extract first two dimensions of matrix as coordinates
X = matrix[:, 0]  # First dimension (x-axis)
Y = matrix[:, 1]  # Second dimension (y-axis)
Z = matrix[:, 2]  # Third dimension (y-axis)


# for the triangle (equilateral) - simplex_2
T = np.array([[-1,0],[1,0],[0,3**(1/2)]])


## Print the first samples from the data set with their GW-Barycentric coordinates

n_samples_to_plot = 5

# Create subplots
fig, axes = plt.subplots(1, n_samples_to_plot, figsize=(15, 5))

for idx, i in enumerate(range(n_samples_to_plot)):
    a = X_train[i, :, 2]  # Extract the third column (weights)
    valid_indices = np.where(a != -1)[0]  # Filter out invalid (-1) values
    a = a[valid_indices]
    a = a / float(a.sum())  # Normalize weights

    U = X_train[i, valid_indices, :2]  # Extract 2D points
    U = utils.normalize_2Dpointcloud_coordinates(U)  # Normalize coordinates

    axes[idx].scatter(U[:, 0], U[:, 1], s=a * 350)  # Scatter plot with size based on weights
    axes[idx].set_aspect('equal', adjustable='box')
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])
    l = np.array([X[i], Y[i], Z[i]])
    coord = l @ T  # compute weighted combination of triangle vertices
    axes[idx].set_title(f'Assigned label = {int(labels[i])} \n Coordinates ({coord[0]:.2f}, {coord[1]:.2f})')

fig.suptitle("First samples from the data set with their GW-Barycentric Coordinates and Predicted Labels", fontsize=14)
plt.tight_layout()
plt.show()




# Create scatter plot

# Project barycentric coordinates to 2D Euclidean coordinates in the triangle
projected_coords = X[:, None] * T[0] + Y[:, None] * T[1] + Z[:, None] * T[2]

# Create figure and scatter plot
plt.figure(figsize=(8, 7))

custom_cmap = ListedColormap(["blue", "red", "green"])
scatter = plt.scatter(projected_coords[:, 0], projected_coords[:, 1], c=labels, cmap=custom_cmap, edgecolors="k", alpha=0.7)

# Draw triangle outline
plt.plot([-1, 0, 1, -1], [0, np.sqrt(3), 0, 0], 'k-', linewidth=1)

# # # Add triangle vertex labels
# # vertex_labels = ['(-1, 0)', '(1, 0)', '(0, √3)']
# for i, txt in enumerate(vertex_labels):
#     plt.text(T[i, 0], T[i, 1] + 0.1, txt, ha='center', fontsize=10)

# Legend
legend_labels = {0: f'Class Digit {selected_digits[0]}',
                 1: f'Class Digit {selected_digits[1]}',
                 2: f'Class Digit {selected_digits[2]}'}
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=6, label=legend_labels[l])
           for l, color in zip([0, 1, 2], ["blue", "red", "green"])]
plt.legend(handles=handles)

# Labels and title
plt.title("GW-Barycentric Coordinates Projected into Triangle (3-class)")
plt.axis('equal')
plt.xticks([])
plt.yticks([])

plt.show()