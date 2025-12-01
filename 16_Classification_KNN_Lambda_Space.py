## This notebook uses the 2D point cloud MNIST dataset.
##
## When prompted, the user must enter "1" for the fixed-point algorithm
## or "2" for the gradient-based algorithm using blow-ups, and press enter.
##
## KNN is applied in the GW coordinate space

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, MDS

## IMPORT USER DEFINED LIBRARIES ##################################################################
import utils


## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [point_coordinate[0],point_coordinate[1],point_mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()


# Select some digits
selected_digits = [0,2, 4]
selected_indices = np.concatenate([digit_indices[d] for d in selected_digits])

# Filter the dataset: pick n_per_class samples per selected digit
n_per_class = 20
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
n_temp = 1  # Number of templates for each digit
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
    else:
        raise ValueError("Method must be 1 (fixed-point) or 2 (gradient via blow-up).")

    # Store label + lambdas
    sample_label = label_selected[i]
    all_lambda_matrix_list.append(np.concatenate(([sample_label], lambdas)))

    # Progress print
    if i % 50 == 0:
        print(f"Processed {i} samples...")

all_lambda_matrix = np.array(all_lambda_matrix_list)

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

print("Template λ-coordinates computed with shape:", template_lambdas.shape)


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

print(f"K-Means accuracy (after optimal label matching): {clust_acc:.4f}")
print("Confusion matrix (true vs mapped cluster labels):")
print(clust_cm)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")


###################################################################################################
## TRAIN/TEST SPLIT ###############################################################################
###################################################################################################
train_fraction = 0.7

X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(
    all_features,
    all_labels,
    train_size=train_fraction,
    random_state=42,
    stratify=all_labels,
)

print("\nTrain size:", X_train_feat.shape[0])
print("Test size:", X_test_feat.shape[0])


###################################################################################################
## K-NN CLASSIFICATION ON GW-BARYCENTRIC COORDINATES ##############################################
###################################################################################################
print("\nK-NN classification on GW-barycentric coordinates")

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_feat, y_train_feat)

# Predict on test set
y_pred = knn.predict(X_test_feat)

# Evaluate
acc = accuracy_score(y_test_feat, y_pred)
cm = confusion_matrix(y_test_feat, y_pred)

print(f"k-NN accuracy (k={k}): {acc:.4f}")
print("Confusion matrix:")
print(cm)
print("\nClassification report:")
print(classification_report(y_test_feat, y_pred))


###################################################################################################
## VISUALIZATION OF GW-BARYCENTRIC COORDINATES ####################################################
###################################################################################################
print("\nVisualizing GW-barycentric coordinates...")

S = n_classes * n_temp          # total number of templates
simplex_dim = S - 1             # intrinsic dimension of the simplex

print("Total number of templates =", S)
print("Intrinsic simplex dimension =", simplex_dim)

# Stack data features and template features together for joint embedding
all_with_templates = np.vstack([all_features, template_lambdas])
n_data = all_features.shape[0]
n_total = all_with_templates.shape[0]


if simplex_dim == 1:
    # ============================================================
    # TRUE 1D SIMPLEX CASE (e.g. 2 templates)
    # ============================================================
    lambdas_1d = all_features[:, 0]   # project from R^2 → R (taking first coord)
    unique_labels = np.unique(all_labels)

    plt.figure(figsize=(7, 5))

    # Number of bins
    bins = (2 * n_classes * n_per_class) // 100
    bins = max(bins, 5)

    # Compute common bin edges from all data
    lmin, lmax = lambdas_1d.min(), lambdas_1d.max()
    bin_edges = np.linspace(lmin, lmax, bins + 1)

    for c in unique_labels:
        mask = (all_labels == c)
        plt.hist(
            lambdas_1d[mask],
            bins=bin_edges,  # common bin edges for all classes
            alpha=0.5,
            density=True,
            label=f"class {c}",
        )

    # Overlay template λ-coordinates as vertical lines
    template_lambdas_1d = template_lambdas[:, 0]
    y_level = 0.02 * plt.ylim()[1]
    for lam_t in template_lambdas_1d:
        plt.plot(lam_t, y_level, marker="x", color="k", markersize=10, mew=2)

    plt.title("1D Histogram - GW-barycentric Coordinates", fontsize=18)
    plt.xlabel("λ", fontsize=16)
    plt.ylabel("Density", fontsize=16)

    plt.yticks([])
    plt.legend(title="Digits", fontsize=14, title_fontsize=15)

    plt.tight_layout()
    plt.savefig("hist_1d_gw_barycentric.pdf", bbox_inches="tight")
    plt.show()

else:
    # ============================================================
    # TRUE HIGHER-DIMENSIONAL CASE → use t-SNE and MDS
    # ============================================================
    print("Simplex dimension > 1, computing t-SNE embedding...")
    # ---------------------------
    # t-SNE embedding
    # ---------------------------
    n_samples = n_total
    perplexity = min(30, max(5, n_samples // 3), n_samples - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="random",
        random_state=42,
    )
    X_tsne_all = tsne.fit_transform(all_with_templates)

    X_tsne = X_tsne_all[:n_data]
    X_tsne_templates = X_tsne_all[n_data:]

    # ------------------------------------------------------------------
    # t-SNE colored by TRUE labels, with templates
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=all_labels,
        cmap="tab10",
        alpha=0.8,
    )

    template_scatter = plt.scatter(
        X_tsne_templates[:, 0],
        X_tsne_templates[:, 1],
        c="black",
        marker="x",
        s=130,
        linewidths=1.5,
        label="Templates",
    )

    plt.title("t-SNE of GW-barycentric coordinates", fontsize=18)
    plt.xlabel("t-SNE Component 1", fontsize=16)
    plt.ylabel("t-SNE Component 2", fontsize=16)
    plt.xticks([])
    plt.yticks([])

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
    plt.savefig("tsne_gw_barycentric.pdf", bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------------
    # t-SNE colored by K-Means CLUSTERS (unsupervised), with templates
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 6))

    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=cluster_labels,  # raw cluster IDs
        cmap="tab10",
        alpha=0.8,
    )

    template_scatter = plt.scatter(
        X_tsne_templates[:, 0],
        X_tsne_templates[:, 1],
        c="black",
        marker="x",
        s=130,
        linewidths=1.5,
        label="Templates",
    )

    plt.title("K-Means Clustering in t-SNE space", fontsize=18)
    plt.xlabel("t-SNE Component 1", fontsize=16)
    plt.ylabel("t-SNE Component 2", fontsize=16)
    plt.xticks([])
    plt.yticks([])

    handles, _ = scatter.legend_elements()
    cluster_names = [f"Class {chr(ord('A') + i)}" for i in range(num_clusters)]
    labels_text = cluster_names + ["Templates"]

    handles.append(template_scatter)

    plt.legend(
        handles,
        labels_text,
        fontsize=14,
        title_fontsize=15,
    )

    plt.tight_layout()
    plt.savefig("tsne_kmeans_clusters.pdf", bbox_inches="tight")
    plt.show()

print("Computing MDS embedding...")
# ---------------------------
# MDS embedding
# ---------------------------
mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42)
X_mds_all = mds.fit_transform(all_with_templates)

X_mds = X_mds_all[:n_data]
X_mds_templates = X_mds_all[n_data:]

# ------------------------------------------------------------------
# MDS colored by TRUE labels, with templates
# ------------------------------------------------------------------
plt.figure(figsize=(7, 6))
scatter = plt.scatter(
    X_mds[:, 0],
    X_mds[:, 1],
    c=all_labels,
    cmap="tab10",
    alpha=0.8,
)

template_scatter = plt.scatter(
    X_mds_templates[:, 0],
    X_mds_templates[:, 1],
    c="black",
    marker="x",
    s=130,
    linewidths=1.5,
    label="Templates",
)

plt.title("MDS of GW-barycentric coordinates", fontsize=18)
plt.xlabel("MDS Component 1", fontsize=16)
plt.ylabel("MDS Component 2", fontsize=16)

plt.xticks([])
plt.yticks([])

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
plt.savefig("mds_gw_barycentric.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# MDS colored by K-Means CLUSTERS (unsupervised), with templates
# ------------------------------------------------------------------
plt.figure(figsize=(7, 6))

scatter = plt.scatter(
    X_mds[:, 0],
    X_mds[:, 1],
    c=cluster_labels,  # raw cluster IDs
    cmap="tab10",
    alpha=0.8,
)

template_scatter = plt.scatter(
    X_mds_templates[:, 0],
    X_mds_templates[:, 1],
    c="black",
    marker="x",
    s=130,
    linewidths=1.5,
    label="Templates",
)

plt.title("K-Means Clustering in MDS space", fontsize=18)
plt.xlabel("MDS Component 1", fontsize=16)
plt.ylabel("MDS Component 2", fontsize=16)
plt.xticks([])
plt.yticks([])

handles, _ = scatter.legend_elements()
cluster_names = [f"Class {chr(ord('A') + i)}" for i in range(num_clusters)]
labels_text = cluster_names + ["Templates"]
handles.append(template_scatter)

plt.legend(
    handles,
    labels_text,
    fontsize=14,
    title_fontsize=15,
)

plt.tight_layout()
plt.savefig("mds_kmeans_clusters.pdf", bbox_inches="tight")
plt.show()
