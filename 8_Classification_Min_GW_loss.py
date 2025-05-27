## MINIMUM GW-BARYCENTER LOSS FOR CLASSIFICATION:
## Classification by applying GW-barycenter analysis and
## using GW-barycenter synthesis loss for determining the class.
## Specifically, let n be the number of classes in a data set.
## Select S random points of each class.
## Given an input, compute the GW-barycentric coordinates
## using the references from each class SEPARATELY. (Analysis)
## Synthesize the corresponding GW-barycenters
## for the different barycenter coordinates obtained before and the respective templates.
## Compute GW-distances between input and synthesize GW-barycenters.
## Select the class with the minimum GW-distance between input
## Compute accuracy between the true labels and the predicted ones.

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split

import ot
from sympy.codegen import Print

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()


## GETTING RANDOM TEMPLATES FROM DATASET ##########################################################
# Templates are of the form (matrix, measure)
n_classes = 10  # Since we are working with digits 0-9
n_temp = 5  # Number of templates for each digit
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





##TEST ONE SAMPLE
print('Testing  in one sample of the data set:')
print(f'As templates we are using {n_temp} random samples of digit point-clouds from 0 to 9.')
print('We compute the GW-barycentric coordinates separately for each class')
print('and, with those coordinates, we synthesize GW-barycenters (as many as classes)')

# Select a random sample
u = np.random.randint(0, 100)
a = Data[u, :, 2]  # Original mass values
X = Data[u, a != -1, :2]  # Extract valid points

# Normalize X
X = X - X.mean(0)[np.newaxis, :]
dist_matrix_input = sp.spatial.distance.cdist(X, X)

# Filter `a` (only keep valid entries)
a = a[a != -1]
a = a / float(a.sum())  # Normalize to sum to 1

print('label of input = ', label[u])

## PERFORM GW-BARYCENTER ANALYSIS FOR EACH CLASS SEPARATELY #######################################
gw_dist_list = [] # to store the GW-distance loos

for j in range(n_classes):
    start_idx = j * n_temp
    end_idx = (j + 1) * n_temp

    # Select the relevant chunk for this iteration
    matrix_chunk = matrix_temp_list[start_idx:end_idx]
    measure_chunk = measure_temp_list[start_idx:end_idx]

    # Compute barycenter coordinates (lambdas) and synthesize new GW-barycenters
    bary, lambdas = utils.get_lambdas(matrix_chunk, measure_chunk, dist_matrix_input, a)

    ## Compare GW-distance between initial and synthesized GW-barycenters
    gromov_distance = ot.gromov.gromov_wasserstein(dist_matrix_input, bary, a, a, log=True)[1]
    gw_dist = gromov_distance['gw_dist']

    # Store results
    gw_dist_list.append(gw_dist)

# Convert list to array
gw_dist_array = np.array(gw_dist_list)

# Predict label based on mim GW-dist
computed_label = np.where(gw_dist_array == gw_dist_array.min())[0][0]
print('min GW-loss class = ',computed_label)

if label[u] == computed_label:
    print('This random input is correctly classified.')
else:
    print('This random input is NOT correctly classified.')



## REPEAT THE ABOVE CLASSIFICATION FOR A SET OF TRAINING SAMPLES FROM THE DATASET #################
print('Now, we will perform the same classification approach for a training set of the data set to finally compute accuracy.')

## GET TRAINING SAMPLES FROM DATASET ##############################################################
# Convert to NumPy arrays
Data = np.array(Data)  # Shape: (num_samples, num_points, 3)
label = np.array(label)  # Shape: (num_samples,)

# Split into training and test sets (test_size% test, (100-test_size)% training)
X_train, X_test, y_train, y_test = train_test_split(Data, label, test_size=0.99, random_state=42, stratify=label)

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



## CLASSIFICATION IN THE TRAINING SET
positive_case = 0 # To compute accuracy

for i in range(len(train_distance_matrices)):
    B = train_distance_matrices[i]  # Distance matrix
    b = train_measures[i]  # Measure
    label = y_train[i]  # Label of the training point

    for j in range(n_classes):
        start_idx = j * n_temp
        end_idx = (j + 1) * n_temp

        # Select the relevant chunk for this iteration
        matrix_chunk = matrix_temp_list[start_idx:end_idx]
        measure_chunk = measure_temp_list[start_idx:end_idx]

        # Compute barycenter coordinates (lambdas) and synthesize new GW-barycenters
        bary, lambdas = utils.get_lambdas(matrix_chunk, measure_chunk, B, b)

        ## Compare GW-distance between initial and synthesized GW-barycenters
        gromov_distance = ot.gromov.gromov_wasserstein(B, bary, b, b, log=True)[1]
        gw_dist = gromov_distance['gw_dist']

        # Store results
        gw_dist_list.append(gw_dist)

    # Convert list to array
    gw_dist_array = np.array(gw_dist_list)

    # Predict label based on mim GW-dist
    computed_label = np.where(gw_dist_array == gw_dist_array.min())[0][0]
    if label == computed_label:
        positive_case += 1

    # Print progress every 50 iterations
    if i % 50 == 0:
        print(f"Processed {i} samples...")
        if i != 0:
            print(f'So far, the accuracy of the method is {positive_case / i}')

print('Accuracy = ', positive_case / len(train_distance_matrices))




