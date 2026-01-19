## This notebook uses the 2D point cloud MNIST dataset.
##
## When prompted, the user must enter "1" for the fixed-point algorithm
## or "2" for the gradient-based algorithm using blow-ups, and press enter.
##
## K-Means is applied in the GW coordinate space

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, MDS
import time


## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils







def barycentric_to_cartesian(L):
    """
    When considering 3 Templates.
    Map barycentric coordinates L (n,3) in Δ² to 2D Cartesian coordinates
    in an equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    T = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2.0],
    ])  # shape (3,2)
    return L @ T  # (n,3) @ (3,2) -> (n,2)




## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()


# Select some digits
selected_digits = [0,4]
selected_indices = np.concatenate([digit_indices[d] for d in selected_digits])

# Filter the dataset: pick n_per_class samples per selected digit
n_per_class = 250
reduced_indices = []

for d in selected_digits:
    inds = digit_indices[d]
    chosen = np.random.choice(inds, n_per_class, replace=False)
    reduced_indices.append(chosen)

reduced_indices = np.concatenate(reduced_indices)

Data_selected = Data[reduced_indices]
label_selected = label[reduced_indices]

print("Reduced full dataset size:", Data_selected.shape[0])


## GETTING RANDOM TEMPLATES FROM DATASET ##########################################################
# Templates are of the form (matrix, measure)
n_classes = len(selected_digits)
n_temp = 6  # Number of templates for each digit
ind_temp_list = []       # list of template indices from dataset
measure_temp_list = []   # list of template measures
matrix_temp_list = []    # list of template dissimilarity matrices

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

print("Random templates, extracted")


## PLOT TEMPLATES #################################################################################
fig, axes = plt.subplots(1, n_classes * n_temp, figsize=(8, 4))
axes = axes.flatten()

for i, ind in enumerate(ind_temp_list):
    a = Data[ind, :, 2]
    X = Data[ind, a != -1, :2]
    X = utils.normalize_2Dpointcloud_coordinates(X)
    a = a[a != -1]
    a = a / float(a.sum())
    axes[i].scatter(X[:, 0], X[:, 1], s=a * 250)
    axes[i].set_aspect("equal", adjustable="box")
    axes[i].set_xticks([])  # Remove x-axis ticks
    axes[i].set_yticks([])  # Remove y-axis ticks

fig.suptitle("Templates", fontsize=24)
plt.tight_layout()
plt.savefig("templates.pdf", bbox_inches="tight")
plt.show()


###################################################################################################
## COMPUTE GW-BARYCENTRIC COORDINATES (lambdas) FOR ALL SELECTED SAMPLES ##########################
###################################################################################################
print("Compute GW-barycentric coordinates for all selected samples from data set")

# Ask the user to choose a method (same for all samples)
method = int(
    input(
        "Choose a method: fixed-point approach [enter 1] or gradient approach via blow-up [enter 2] "
    )
)

all_lambda_matrix_list = []

average_size = 0

# --- timing: feature extraction for λ-method ---
t0_lambdas = time.perf_counter()

for i in range(Data_selected.shape[0]):
    # Extract probability measure (third column) and filter valid points
    p_s = Data_selected[i, :, 2]
    valid_indices = np.where(p_s != -1)[0]
    p_s = p_s[valid_indices]

    # Normalize p_s to make it a valid probability distribution
    p_s = p_s / float(p_s.sum())

    # Extract and normalize spatial coordinates (first two columns)
    C_s = Data_selected[i, valid_indices, :2]
    C_s = utils.normalize_2Dpointcloud_coordinates(C_s)

    # Compute pairwise Euclidean distance matrix
    dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)

    # Compute lambdas using the chosen method
    if method == 1:
        # _, lambdas = utils.get_lambdas(matrix_temp_list, measure_temp_list, dist_matrix_s, p_s)
        _, lambdas = utils.get_lambdas_constraints(
            matrix_temp_list, measure_temp_list, dist_matrix_s, p_s
        )
    elif method == 2:
        dist_matrix_s_bu, p_s_bu, temp_blow_up = utils.blow_up(
            matrix_temp_list, measure_temp_list, dist_matrix_s, p_s
        )
        _, lambdas = utils.get_lambdas_blowup(temp_blow_up, dist_matrix_s_bu, p_s_bu)

        average_size += dist_matrix_s_bu.shape[0]

    else:
        raise ValueError("Method must be 1 (fixed-point) or 2 (gradient via blow-up).")
    # Store label + lambdas
    sample_label = label_selected[i]
    all_lambda_matrix_list.append(np.concatenate(([sample_label], lambdas)))

    # Progress print
    if i % 50 == 0:
        print(f"Processed {i} samples...")


all_lambda_matrix = np.array(all_lambda_matrix_list)

time_lambdas = time.perf_counter() - t0_lambdas  # total time to compute all λ's

print(f'Average blow-up size: {average_size/Data_selected.shape[0]}')
print(f"Time to compute all GW-barycentric coordinates: {time_lambdas:.4f} s")

print("Barycentric coordinates for all samples, computed")


# First column: labels, rest: GW-barycentric coordinates
all_labels = all_lambda_matrix[:, 0].astype(int)
all_features = all_lambda_matrix[:, 1:]


################################################################################
# COMPUTE GW-BARYCENTRIC COORDINATES FOR TEMPLATES THEMSELVES ##################
################################################################################
print("\nCompute GW-barycentric coordinates for templates")

template_lambda_list = []
template_label_list = []

# Each template in matrix_temp_list / measure_temp_list corresponds to a digit in selected_digits
# repeated n_temp times and in the same order as the template-building loop.
template_digit_labels = []
for d in selected_digits:
    for _ in range(n_temp):
        template_digit_labels.append(d)
template_digit_labels = np.array(template_digit_labels)

for B_temp, b_temp, lab in zip(matrix_temp_list, measure_temp_list, template_digit_labels):
    if method == 1:
        _, lambdas_temp = utils.get_lambdas_constraints(
            matrix_temp_list, measure_temp_list, B_temp, b_temp
        )
    elif method == 2:
        B_bu, b_bu, temp_blow_up = utils.blow_up(
            matrix_temp_list, measure_temp_list, B_temp, b_temp
        )
        _, lambdas_temp = utils.get_lambdas_blowup(temp_blow_up, B_bu, b_bu)
    else:
        raise ValueError("Method must be 1 or 2.")

    template_lambda_list.append(lambdas_temp)
    template_label_list.append(lab)

template_lambdas = np.array(template_lambda_list)  # shape: (S, dim)
template_labels = np.array(template_label_list)    # digit label of each template




###################################################################################################
## K-MEANS CLUSTERING ON λ-SPACE #################################################################
###################################################################################################
print("\nK-Means clustering on GW-barycentric coordinates")

# Number of clusters = number of classes
num_clusters = n_classes   # e.g. len(selected_digits)

# Fit K-Means on ALL samples in λ-space (data only)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(all_features)   # shape: (n_samples,)

# ------------------------------------------------------------------
# Optimal label matching (Hungarian algorithm)
# Map each cluster ID -> best matching true label
# ------------------------------------------------------------------
unique_true = np.unique(all_labels)

# Build confusion matrix between clusters and true labels
# rows = clusters, cols = true labels
cost_matrix = np.zeros((num_clusters, len(unique_true)), dtype=int)
for c in range(num_clusters):
    for j, lab_val in enumerate(unique_true):
        # number of points in cluster c with true label lab_val
        cost_matrix[c, j] = np.sum((cluster_labels == c) & (all_labels == lab_val))

# Hungarian algorithm on NEGATIVE counts to maximize agreement
row_ind, col_ind = sp.optimize.linear_sum_assignment(-cost_matrix)

# cluster_id -> matched label
cluster_to_label = {}
for r, c in zip(row_ind, col_ind):
    cluster_id = r
    label_id = unique_true[c]
    cluster_to_label[cluster_id] = label_id

# Apply mapping to all cluster assignments
mapped_labels = np.array([cluster_to_label[c] for c in cluster_labels])

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
clust_acc = accuracy_score(all_labels, mapped_labels)
clust_cm = confusion_matrix(all_labels, mapped_labels, labels=unique_true)
ari = adjusted_rand_score(all_labels, cluster_labels)  # uses original cluster ids
nmi = normalized_mutual_info_score(all_labels, cluster_labels)

# print(f"K-Means accuracy: {clust_acc:.4f}")
print("Confusion matrix (true vs mapped cluster labels):")
print(clust_cm)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")


###################################################################################################

print("\nVisualizing λ in the simplex with K-Means decision regions")

# Ensure we are really in Δ²: 3 templates
if all_features.shape[1] == 3:
    # Normalize λ's to sum to 1 (for numerical stability)
    L_data = all_features / all_features.sum(axis=1, keepdims=True)
    L_temp = template_lambdas / template_lambdas.sum(axis=1, keepdims=True)

    # Project to 2D triangle
    XY_data = barycentric_to_cartesian(L_data)
    XY_temp = barycentric_to_cartesian(L_temp)

    # ------------------------------------------------------------------
    # Build a grid in Δ² and get K-Means decision regions in λ-space
    # ------------------------------------------------------------------
    n_grid = 200  # resolution of the grid in the simplex
    grid_bary = []

    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            l1 = i / n_grid
            l2 = j / n_grid
            l3 = 1.0 - l1 - l2
            grid_bary.append([l1, l2, l3])

    grid_bary = np.array(grid_bary)  # (N_grid, 3)
    grid_xy = barycentric_to_cartesian(grid_bary)

    # Predict cluster for each barycentric grid point using K-Means in λ-space
    grid_clusters = kmeans.predict(grid_bary)

    # ------------------------------------------------------------------
    # Plot: decision regions + data (true labels) + templates
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 6))

    # 1) K-Means decision regions (soft background)
    plt.scatter(
        grid_xy[:, 0],
        grid_xy[:, 1],
        c=grid_clusters,
        cmap="tab10",
        alpha=0.15,
        s=10,
        linewidths=0,
    )

    # 2) Data points, colored by TRUE labels
    scatter = plt.scatter(
        XY_data[:, 0],
        XY_data[:, 1],
        c=all_labels,
        cmap="tab10",
        alpha=0.8,
        edgecolors="k",
        linewidths=0.3,
    )

    # 3) Templates as black crosses
    template_scatter = plt.scatter(
        XY_temp[:, 0],
        XY_temp[:, 1],
        c="black",
        marker="x",
        s=130,
        linewidths=1.5,
        label="Templates",
    )

    # 4) Draw triangle border of Δ²
    verts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2.0],
        [0.0, 0.0],
    ])
    plt.plot(verts[:, 0], verts[:, 1], "k-", linewidth=1.5)

    plt.title("K-Means decision regions in the simplex", fontsize=18)
    plt.xlabel("Simplex coordinate 1", fontsize=16)
    plt.ylabel("Simplex coordinate 2", fontsize=16)
    plt.xticks([])
    plt.yticks([])

    # Legend: classes + templates
    handles, _ = scatter.legend_elements()
    class_labels = sorted(np.unique(all_labels))
    labels_text = [f"Class {c}" for c in class_labels]
    handles.append(template_scatter)
    labels_text.append("Templates")

    plt.legend(
        handles,
        labels_text,
        loc="best",
        fontsize=14,
        title_fontsize=15,
    )

    plt.tight_layout()
    plt.savefig("simplex_lambda_kmeans_decision_regions.pdf", bbox_inches="tight")
    plt.show()


