## This notebook uses the 2D point cloud MNIST dataset.
##
## We compare:
##  (1) GW-barycentric λ (fixed-point) + k-NN
##  (2) GW-barycentric λ (blow-up / gradient) + k-NN
##  where in (1) and (2) KNN is applied in the GW coordinate (feature) space
##  (3) GW 1-NN (templates, direct GW)
##  (4) GW dictionary learning + k-NN on unmixing weights
##  by C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online Graph Dictionary Learning (OGDL), ICML, 2021.
##  where KNN is applied in the GW coordinate (feature) space
##
## Additional visualizations are provided


import ot

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    adjusted_rand_score, # ARI <--- used for K-Means
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
    Map barycentric coordinates in the simplex, represented as
    an equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    T = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2.0],
    ])  # shape (3,2)
    return L @ T  # (n,3) @ (3,2) -> (n,2)


def repeated_cv_knn(features, labels, candidate_ks, n_repeats=20, train_fraction=0.7):
    """
    Run repeated train/test splits with inner cross validation (CV) to select k in k-NN.

    Parameters
    ----------
    features : ndarray (n_samples, d)
        Feature matrix.
    labels : ndarray (n_samples,)
        Class labels.
    candidate_ks : list[int]
        Candidate k values for k-NN.
    n_repeats : int
        Number of outer repetitions.
    train_fraction : float
        Fraction of data used for training in each outer split.

    Returns
    -------
    all_test_acc : np.ndarray of shape (n_repeats,)
        Test accuracies for each outer run.
    all_best_ks : np.ndarray of shape (n_repeats,)
        Chosen k for each outer run.
    total_time : float
        Wall-clock time for the whole procedure.
    """
    all_test_acc = []
    all_best_ks = []

    t0 = time.perf_counter()

    for run in range(n_repeats):
        print(f"\n=== Outer run {run+1}/{n_repeats} ===")

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            train_size=train_fraction,
            random_state=run,
            stratify=labels,
        )

        print("  Train size:", X_train.shape[0])
        print("  Test size:", X_test.shape[0])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)

        mean_cv_scores = []
        for k in candidate_ks:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train, y_train, cv=cv)
            mean_score = scores.mean()
            mean_cv_scores.append(mean_score)
            print(f"  k = {k}: CV accuracy = {mean_score:.4f}")

        best_idx = int(np.argmax(mean_cv_scores))
        best_k = candidate_ks[best_idx]
        all_best_ks.append(best_k)
        print(f"  -> Best k on this run = {best_k}")

        knn_best = KNeighborsClassifier(n_neighbors=best_k)
        knn_best.fit(X_train, y_train)
        y_pred = knn_best.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        all_test_acc.append(test_acc)

        print(f"  Test accuracy (best k={best_k}): {test_acc:.4f}")
        print("  Confusion matrix (true vs predicted):")
        print(confusion_matrix(y_test, y_pred, labels=np.unique(labels)))
        print("\n  Classification report:")
        print(classification_report(y_test, y_pred, digits=4))

    total_time = time.perf_counter() - t0

    return np.array(all_test_acc), np.array(all_best_ks), total_time


## DATASET LOADING ################################################################################
# Data: array of the form (sample_index, point_index, [x, y, mass])
# label: labels(0-9) for Data
# digit_indices: list(len 10) of indices for each digit (0-9)
Data, label, digit_indices = utils.load_pointcloudmnist2d()

# Select some digits
selected_digits = [0, 4]
selected_indices = np.concatenate([digit_indices[d] for d in selected_digits])

# Filter the dataset: pick n_per_class samples per selected digit
n_per_class = 200
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
n_temp = 3  # Number of templates for each digit
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

        # Center and normalize coordinates to fit within the unit square [0,1]²
        C_s = utils.normalize_2Dpointcloud_coordinates(C_s)

        # Compute the pairwise Euclidean distance matrix for C_s
        dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)
        matrix_temp_list.append(dist_matrix_s)

print("Random templates, extracted")

## PLOT TEMPLATES #################################################################################
## OPTIONAL:
# fig, axes = plt.subplots(1, n_classes * n_temp, figsize=(8, 4))
# axes = axes.flatten()
#
# for i, ind in enumerate(ind_temp_list):
#     a = Data[ind, :, 2]
#     X = Data[ind, a != -1, :2]
#     X = utils.normalize_2Dpointcloud_coordinates(X)
#     a = a[a != -1]
#     a = a / float(a.sum())
#     axes[i].scatter(X[:, 0], X[:, 1], s=a * 250)
#     axes[i].set_aspect("equal", adjustable="box")
#     axes[i].set_xticks([])  # Remove x-axis ticks
#     axes[i].set_yticks([])  # Remove y-axis ticks
#
# fig.suptitle("Templates", fontsize=24)
# plt.tight_layout()
# plt.savefig("templates.pdf", bbox_inches="tight")
# plt.show()


###################################################################################################
## PRE-PROCESS DATA TO GET DISTANCE MATRICES FOR EACH POINT CLOUD #################################
###################################################################################################

dist_matrices = []
measures = []
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

    # Store it
    dist_matrices.append(dist_matrix_s)
    measures.append(p_s)


###################################################################################################
## COMPUTE GW-BARYCENTRIC COORDINATES (lambdas) FOR ALL SELECTED SAMPLES ##########################
## We compute BOTH: fixed-point (FP) and gradient via blow-up (BU) ################################
###################################################################################################
print("Compute GW-barycentric coordinates for all selected samples from data set")
print("We will compute GW coordinates with both methods: (1) fixed-point and (2) blow-up.\n")

all_lambda_fp_list = []    # [label | λ_fp]
all_lambda_bu_list = []    # [label | λ_bu]

average_bu_size = 0.0

# --- fixed-point method -------------------------------------------------------------------------
print(">>> Computing λ via fixed-point method for all samples...")
t0_lambdas_fp = time.perf_counter()

for i in range(Data_selected.shape[0]):
    # Fixed-point λ
    _, lambdas_fp = utils.get_lambdas_constraints(
        matrix_temp_list, measure_temp_list, dist_matrices[i], measures[i]
    )

    sample_label = label_selected[i]
    all_lambda_fp_list.append(np.concatenate(([sample_label], lambdas_fp)))

    if i % 50 == 0:
        print(f"  [FP] Processed {i} samples...")

time_lambdas_fp = time.perf_counter() - t0_lambdas_fp
print(f"Time to compute all GW-barycentric coordinates (fixed-point): {time_lambdas_fp:.4f} s")

# --- gradient via blow-up method ----------------------------------------------------------------
print("\n>>> Computing λ via gradient method (blow-up) for all samples...")
t0_lambdas_bu = time.perf_counter()

for i in range(Data_selected.shape[0]):
    # Blow-up + λ estimation
    dist_matrix_s_bu, p_s_bu, temp_blow_up = utils.blow_up(
        matrix_temp_list, measure_temp_list, dist_matrices[i], measures[i]
    )
    _, lambdas_bu = utils.get_lambdas_blowup(temp_blow_up, dist_matrix_s_bu, p_s_bu)

    average_bu_size += dist_matrix_s_bu.shape[0]

    sample_label = label_selected[i]
    all_lambda_bu_list.append(np.concatenate(([sample_label], lambdas_bu)))

    if i % 50 == 0:
        print(f"  [BU] Processed {i} samples...")

time_lambdas_bu = time.perf_counter() - t0_lambdas_bu
average_bu_size /= Data_selected.shape[0]


print(f"Average blow-up size: {average_bu_size}")
print(f"Time to compute all GW-barycentric coordinates (blow-up): {time_lambdas_bu:.4f} s")
print("Barycentric coordinates for all samples, computed (both methods).")

# Convert to arrays
all_lambda_fp = np.array(all_lambda_fp_list)
all_lambda_bu = np.array(all_lambda_bu_list)

# Labels are the same for both methods
all_labels = all_lambda_fp[:, 0].astype(int)

# Features (λ) for each method
all_features_fp = all_lambda_fp[:, 1:]
all_features_bu = all_lambda_bu[:, 1:]



###################################################################################################
## ANALYSIS IN GW COORDINATE SPACE ################################################################
## K NN and Visualizations ########################################################################
###################################################################################################



# ###################################################################################################
# ## K-MEANS CLUSTERING ON λ-SPACE (using FP features) ##############################################
# ###################################################################################################
# print("\nK-Means clustering on GW-barycentric coordinates (fixed-point λ)")
#
# # Number of clusters = number of classes
# num_clusters = n_classes   # e.g. len(selected_digits)
#
# # Fit K-Means on ALL samples in λ-space (data only)
# kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
# cluster_labels = kmeans.fit_predict(all_features)   # shape: (n_samples,)
#
# # ------------------------------------------------------------------
# # Optimal label matching (Hungarian algorithm)
# # ------------------------------------------------------------------
# unique_true = np.unique(all_labels)
#
# # Build confusion matrix between clusters and true labels
# # rows = clusters, cols = true labels
# cost_matrix = np.zeros((num_clusters, len(unique_true)), dtype=int)
# for c in range(num_clusters):
#     for j, lab_val in enumerate(unique_true):
#         # number of points in cluster c with true label lab_val
#         cost_matrix[c, j] = np.sum((cluster_labels == c) & (all_labels == lab_val))
#
# # Hungarian algorithm on NEGATIVE counts to maximize agreement
# row_ind, col_ind = sp.optimize.linear_sum_assignment(-cost_matrix)
#
# # cluster_id -> matched label
# cluster_to_label = {}
# for r, c in zip(row_ind, col_ind):
#     cluster_id = r
#     label_id = unique_true[c]
#     cluster_to_label[cluster_id] = label_id
#
# # Apply mapping to all cluster assignments
# mapped_labels = np.array([cluster_to_label[c] for c in cluster_labels])
#
# # ------------------------------------------------------------------
# # Evaluation
# # ------------------------------------------------------------------
# clust_acc = accuracy_score(all_labels, mapped_labels)
# clust_cm = confusion_matrix(all_labels, mapped_labels, labels=unique_true)
# ari = adjusted_rand_score(all_labels, cluster_labels)  # uses original cluster ids
# nmi = normalized_mutual_info_score(all_labels, cluster_labels)
#
# print(f"K-Means accuracy: {clust_acc:.4f}")
# print("Confusion matrix (true vs mapped cluster labels):")
# print(clust_cm)
# print(f"Adjusted Rand Index (ARI): {ari:.4f}")
# print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
#
# ###################################################################################################
#
# print("\nVisualizing λ in the simplex with K-Means decision regions (FP λ)")
#
# # Ensure we are really in SIMPLEX: 3 templates
# if all_features.shape[1] == 3:
#     # Normalize λ's to sum to 1 (for numerical stability)
#     L_data = all_features / all_features.sum(axis=1, keepdims=True)
#     L_temp = template_lambdas / template_lambdas.sum(axis=1, keepdims=True)
#
#     # Project to 2D triangle
#     XY_data = barycentric_to_cartesian(L_data)
#     XY_temp = barycentric_to_cartesian(L_temp)
#
#     # ------------------------------------------------------------------
#     # Build a grid in SIMPLEX and get K-Means decision regions in λ-space
#     # ------------------------------------------------------------------
#     n_grid = 200  # resolution of the grid in the simplex
#     grid_bary = []
#
#     for i in range(n_grid + 1):
#         for j in range(n_grid + 1 - i):
#             l1 = i / n_grid
#             l2 = j / n_grid
#             l3 = 1.0 - l1 - l2
#             grid_bary.append([l1, l2, l3])
#
#     grid_bary = np.array(grid_bary)  # (N_grid, 3)
#     grid_xy = barycentric_to_cartesian(grid_bary)
#
#     # Predict cluster for each barycentric grid point using K-Means in λ-space
#     grid_clusters = kmeans.predict(grid_bary)
#
#     # ------------------------------------------------------------------
#     # Plot: decision regions + data (true labels) + templates
#     # ------------------------------------------------------------------
#     plt.figure(figsize=(7, 6))
#
#     # 1) K-Means decision regions (soft background)
#     plt.scatter(
#         grid_xy[:, 0],
#         grid_xy[:, 1],
#         c=grid_clusters,
#         cmap="tab10",
#         alpha=0.15,
#         s=10,
#         linewidths=0,
#     )
#
#     # 2) Data points, colored by TRUE labels
#     scatter = plt.scatter(
#         XY_data[:, 0],
#         XY_data[:, 1],
#         c=all_labels,
#         cmap="tab10",
#         alpha=0.8,
#         edgecolors="k",
#         linewidths=0.3,
#     )
#
#     # 3) Templates as black crosses
#     template_scatter = plt.scatter(
#         XY_temp[:, 0],
#         XY_temp[:, 1],
#         c="black",
#         marker="x",
#         s=130,
#         linewidths=1.5,
#         label="Templates",
#     )
#
#     # 4) Draw triangle border of SIMPLEX
#     verts = np.array([
#         [0.0, 0.0],
#         [1.0, 0.0],
#         [0.5, np.sqrt(3) / 2.0],
#         [0.0, 0.0],
#     ])
#     plt.plot(verts[:, 0], verts[:, 1], "k-", linewidth=1.5)
#
#     plt.title("K-Means decision regions in the simplex (fixed-point λ)", fontsize=18)
#     plt.xlabel("Simplex coordinate 1", fontsize=16)
#     plt.ylabel("Simplex coordinate 2", fontsize=16)
#     plt.xticks([])
#     plt.yticks([])
#
#     # Legend: classes + templates
#     handles, _ = scatter.legend_elements()
#     class_labels = sorted(np.unique(all_labels))
#     labels_text = [f"Class {c}" for c in class_labels]
#     handles.append(template_scatter)
#     labels_text.append("Templates")
#
#     plt.legend(
#         handles,
#         labels_text,
#         loc="best",
#         fontsize=14,
#         title_fontsize=15,
#     )
#
#     plt.tight_layout()
#     plt.savefig("simplex_lambda_kmeans_decision_regions.pdf", bbox_inches="tight")
#     plt.show()


###################################################################################################
## CROSS-VALIDATED K-NN ON GW-BARYCENTRIC COORDINATES (BOTH METHODS) ##############################
###################################################################################################
print("\nCross-validated k-NN on GW-barycentric coordinates (fixed-point and blow-up)")

candidate_ks = [1, 3, 5, 7, 9]
n_repeats = 10
train_fraction = 0.7

# --- Fixed-point λ -------------------------------------------------------------------------------
print("\n>>> λ (fixed-point) + k-NN")
all_test_accuracies_fp, all_best_ks_fp, time_knn_fp = repeated_cv_knn(
    all_features_fp, all_labels, candidate_ks, n_repeats=n_repeats, train_fraction=train_fraction
)

print("\n====================================================")
print("Fixed-point λ: summary over all runs")
print("Per-run test accuracies:")
print(all_test_accuracies_fp)

print(f"\nMean test accuracy (FP) over {n_repeats} runs: {all_test_accuracies_fp.mean():.4f}")
print(f"Std of test accuracy (FP) over {n_repeats} runs: {all_test_accuracies_fp.std():.4f}")
print(f"Total time for FP λ-method k-NN (CV + repeats): {time_knn_fp:.4f} s")

unique_ks_fp, counts_ks_fp = np.unique(all_best_ks_fp, return_counts=True)
print("\nBest-k frequency over runs (FP):")
for k_val, count in zip(unique_ks_fp, counts_ks_fp):
    print(f"  k = {k_val}: chosen {count} times out of {n_repeats}")

# --- Blow-up λ ----------------------------------------------------------------------------------
print("\n>>> λ (blow-up) + k-NN")
all_test_accuracies_bu, all_best_ks_bu, time_knn_bu = repeated_cv_knn(
    all_features_bu, all_labels, candidate_ks, n_repeats=n_repeats, train_fraction=train_fraction
)

print("\n====================================================")
print("Blow-up λ: summary over all runs")
print("Per-run test accuracies:")
print(all_test_accuracies_bu)

print(f"\nMean test accuracy (BU) over {n_repeats} runs: {all_test_accuracies_bu.mean():.4f}")
print(f"Std of test accuracy (BU) over {n_repeats} runs: {all_test_accuracies_bu.std():.4f}")
print(f"Total time for BU λ-method k-NN (CV + repeats): {time_knn_bu:.4f} s")

unique_ks_bu, counts_ks_bu = np.unique(all_best_ks_bu, return_counts=True)
print("\nBest-k frequency over runs (BU):")
for k_val, count in zip(unique_ks_bu, counts_ks_bu):
    print(f"  k = {k_val}: chosen {count} times out of {n_repeats}")



###################################################################################################
## VISUALIZATION OF GW-BARYCENTRIC COORDINATES ####################################################
###################################################################################################

## FP λ ###########################################################################################
# For unsupervised clustering / visualization we use first FP λ
all_features = all_features_fp

################################################################################
# COMPUTE GW-BARYCENTRIC COORDINATES FOR TEMPLATES THEMSELVES ##################
################################################################################
print("\nCompute GW-barycentric coordinates for templates (fixed-point)")

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
    # Template λ's via fixed-point constraints
    _, lambdas_temp = utils.get_lambdas_constraints(
        matrix_temp_list, measure_temp_list, B_temp, b_temp
    )
    template_lambda_list.append(lambdas_temp)
    template_label_list.append(lab)

template_lambdas = np.array(template_lambda_list)  # shape: (S, dim)
template_labels = np.array(template_label_list)    # digit label of each template


print("\nVisualizing GW-barycentric coordinates (fixed-point λ)...")

S = n_classes * n_temp          # total number of templates
simplex_dim = S - 1             # intrinsic dimension of the simplex

print("Total number of templates =", S)
print("Intrinsic simplex dimension =", simplex_dim)

# Stack data features and template features together for joint embedding
all_with_templates = np.vstack([all_features, template_lambdas])
n_data = all_features.shape[0]
n_total = all_with_templates.shape[0]


## For Aesthetics and Comparison purposes -----------------------
lambdas_1d_fp = all_features_fp[:, 1]
lambdas_1d_bu = all_features_bu[:, 1]
unique_labels = np.unique(all_labels)

# Number of bins (same rule you already use)
bins = (2 * n_classes * n_per_class) // 100
bins = max(bins, 5)

# Common bin edges across FP and BU
lmin = min(lambdas_1d_fp.min(), lambdas_1d_bu.min())
lmax = max(lambdas_1d_fp.max(), lambdas_1d_bu.max())
bin_edges = np.linspace(lmin, lmax, bins + 1)

# Global ymax across BOTH methods and ALL classes (with density=True)
global_ymax = 0.0
for c in unique_labels:
    mask = (all_labels == c)

    h_fp, _ = np.histogram(lambdas_1d_fp[mask], bins=bin_edges, density=True)
    h_bu, _ = np.histogram(lambdas_1d_bu[mask], bins=bin_edges, density=True)

    global_ymax = max(global_ymax, h_fp.max(), h_bu.max())

# A little headroom for aesthetics / template markers
global_ylim = (0.0, 1.10 * global_ymax)
##---------------------------------------------------------------


if simplex_dim == 1:   #<---- HISTOGRAM REPRESENTATION
    # ============================================================
    # TRUE 1D SIMPLEX CASE (e.g. 2 templates)
    # ============================================================

    lambdas_1d = all_features[:, 1]

    plt.figure(figsize=(7, 5))

    for c in unique_labels:
        mask = (all_labels == c)
        plt.hist(
            lambdas_1d[mask],
            bins=bin_edges,  # <-- use shared bin edges
            alpha=0.5,
            density=True,
            label=f"class {c}",
        )

    plt.ylim(*global_ylim)  # <-- normalize y-axis across FP and BU

    # Overlay template λ-coordinates
    template_lambdas_1d = template_lambdas[:, 0]
    y_level = 0.02 * global_ylim[1]  # <-- use the fixed y-limit (not plt.ylim())
    for lam_t in template_lambdas_1d:
        plt.plot(lam_t, y_level, marker="x", color="k", markersize=10, mew=2)

    # lambdas_1d = all_features[:, 1]   # project from R^2 → R
    # unique_labels = np.unique(all_labels)
    #
    # plt.figure(figsize=(7, 5))
    #
    # # Number of bins
    # bins = (2 * n_classes * n_per_class) // 100
    # bins = max(bins, 5)
    #
    # # Compute common bin edges from all data
    # lmin, lmax = lambdas_1d.min(), lambdas_1d.max()
    # bin_edges = np.linspace(lmin, lmax, bins + 1)
    #
    # for c in unique_labels:
    #     mask = (all_labels == c)
    #     plt.hist(
    #         lambdas_1d[mask],
    #         bins=bin_edges,  # common bin edges for all classes
    #         alpha=0.5,
    #         density=True,
    #         label=f"class {c}",
    #     )
    #
    # # Overlay template λ-coordinates
    # template_lambdas_1d = template_lambdas[:, 0]
    # y_level = 0.02 * plt.ylim()[1]
    # for lam_t in template_lambdas_1d:
    #     plt.plot(lam_t, y_level, marker="x", color="k", markersize=10, mew=2)

    plt.title("1D Histogram - GW-barycentric Coordinates (FP)", fontsize=18)
    plt.xlabel("λ", fontsize=16)
    plt.ylabel("Density", fontsize=16)

    plt.yticks([])
    plt.legend(title="Digits", fontsize=14, title_fontsize=15)

    plt.tight_layout()
    plt.savefig("hist_1d_gw_barycentric_fp.pdf", bbox_inches="tight")
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

    plt.title("t-SNE of GW-barycentric coordinates (FP λ)", fontsize=18)
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
    plt.savefig("tsne_gw_barycentric_fp.pdf", bbox_inches="tight")
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
    alpha=0.6,
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

plt.title("MDS of GW-barycentric coordinates (FP λ)", fontsize=18)
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
plt.savefig("mds_gw_barycentric_fp.pdf", bbox_inches="tight")
plt.show()


## BU λ ###########################################################################################
# For unsupervised clustering / visualization we use now BU λ
all_features = all_features_bu

################################################################################
# COMPUTE GW-BARYCENTRIC COORDINATES FOR TEMPLATES THEMSELVES ##################
################################################################################
print("\nCompute GW-barycentric coordinates for templates (blow-up)")

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
    # Template λ's via bu

    dist_matrix_temp_bu, p_temp_bu, temp_blow_up = utils.blow_up(
        matrix_temp_list, measure_temp_list, B_temp, b_temp
    )
    _, lambdas_temp = utils.get_lambdas_blowup(temp_blow_up, dist_matrix_temp_bu, p_temp_bu)
    template_lambda_list.append(lambdas_temp)
    template_label_list.append(lab)

template_lambdas = np.array(template_lambda_list)  # shape: (S, dim)
template_labels = np.array(template_label_list)    # digit label of each template


print("\nVisualizing GW-barycentric coordinates (blow-up λ)...")

S = n_classes * n_temp          # total number of templates
simplex_dim = S - 1             # intrinsic dimension of the simplex

print("Total number of templates =", S)
print("Intrinsic simplex dimension =", simplex_dim)

# Stack data features and template features together for joint embedding
all_with_templates = np.vstack([all_features, template_lambdas])
n_data = all_features.shape[0]
n_total = all_with_templates.shape[0]

if simplex_dim == 1:   #<---- HISTOGRAM REPRESENTATION
    # ============================================================
    # TRUE 1D SIMPLEX CASE (e.g. 2 templates)
    # ============================================================

    lambdas_1d = all_features[:, 1]  # (FP or BU depending on the block)

    plt.figure(figsize=(7, 5))

    for c in unique_labels:
        mask = (all_labels == c)
        plt.hist(
            lambdas_1d[mask],
            bins=bin_edges,  # <-- use shared bin edges
            alpha=0.5,
            density=True,
            label=f"class {c}",
        )

    plt.ylim(*global_ylim)  # <-- normalize y-axis across FP and BU

    # Overlay template λ-coordinates
    template_lambdas_1d = template_lambdas[:, 0]
    y_level = 0.02 * global_ylim[1]  # <-- use the fixed y-limit (not plt.ylim())
    for lam_t in template_lambdas_1d:
        plt.plot(lam_t, y_level, marker="x", color="k", markersize=10, mew=2)

    # lambdas_1d = all_features[:, 1]   # project from R^2 → R
    # unique_labels = np.unique(all_labels)
    #
    # plt.figure(figsize=(7, 5))
    #
    # # Number of bins
    # bins = (2 * n_classes * n_per_class) // 100
    # bins = max(bins, 5)
    #
    # # Compute common bin edges from all data
    # lmin, lmax = lambdas_1d.min(), lambdas_1d.max()
    # bin_edges = np.linspace(lmin, lmax, bins + 1)
    #
    # for c in unique_labels:
    #     mask = (all_labels == c)
    #     plt.hist(
    #         lambdas_1d[mask],
    #         bins=bin_edges,  # common bin edges for all classes
    #         alpha=0.5,
    #         density=True,
    #         label=f"class {c}",
    #     )
    #
    # # Overlay template λ-coordinates
    # template_lambdas_1d = template_lambdas[:, 0]
    # y_level = 0.02 * plt.ylim()[1]
    # for lam_t in template_lambdas_1d:
    #     plt.plot(lam_t, y_level, marker="x", color="k", markersize=10, mew=2)

    plt.title("1D Histogram - GW-barycentric Coordinates (BU)", fontsize=18)
    plt.xlabel("λ", fontsize=16)
    plt.ylabel("Density", fontsize=16)

    plt.yticks([])
    plt.legend(title="Digits", fontsize=14, title_fontsize=15)

    plt.tight_layout()
    plt.savefig("hist_1d_gw_barycentric_bu.pdf", bbox_inches="tight")
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

    plt.title("t-SNE of GW-barycentric coordinates (BU λ)", fontsize=18)
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
    plt.savefig("tsne_gw_barycentric_bu.pdf", bbox_inches="tight")
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
    alpha=0.6,
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

plt.title("MDS of GW-barycentric coordinates (BU λ)", fontsize=18)
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
plt.savefig("mds_gw_barycentric_bu.pdf", bbox_inches="tight")
plt.show()

###################################################################################################
###################################################################################################
###################################################################################################


###################################################################################################
## COMPARISON WITH OTHER BASE-LINES: (3) AND (4) ##################################################
###################################################################################################


###################################################################################################
## (3) 1-NN CLASSIFICATION USING DIRECT GW DISTANCE TO TEMPLATES (NO λ-COORDINATES) ###############
###################################################################################################
print("\nRepeated 1-NN classification using direct GW distance to templates (no barycentric coordinates)")

# We classify each sample in Data_selected by:
#   y_pred(i) = label of the template with smallest GW distance to sample i.
# We repeat this experiment with different randomly chosen templates to
# obtain a distribution of accuracies.

n_samples = Data_selected.shape[0]
n_repeats_gw = 5      # number of repetitions
n_temp = 1             # templates per digit

all_gw_accuracies = []

# --- timing: full GW 1-NN baseline over all runs ---
t0_gw1nn = time.perf_counter()

for run in range(n_repeats_gw):
    print(f"\n=== GW 1-NN run {run+1}/{n_repeats_gw} ===")

    # ------------------------------------------------------------------
    # (Re)build random templates for this run
    # ------------------------------------------------------------------
    matrix_temp_list_run = []
    measure_temp_list_run = []
    template_digit_labels_run = []

    for digit in selected_digits:
        for _ in range(n_temp):
            # Select a random index corresponding to the chosen digit
            inds_d = digit_indices[digit]
            ind = inds_d[np.random.randint(len(inds_d))]
            template_digit_labels_run.append(digit)

            # Extract the probability measure from the third column of Data (p_s)
            p_s = Data[ind, :, 2]

            # Valid points
            valid_indices = np.where(p_s != -1)[0]
            p_s = p_s[valid_indices]
            p_s = p_s / float(p_s.sum())  # normalize

            # Extract and normalize spatial coordinates
            C_s = Data[ind, valid_indices, :2]
            C_s = utils.normalize_2Dpointcloud_coordinates(C_s)

            # Distance matrix for template
            dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)

            measure_temp_list_run.append(p_s)
            matrix_temp_list_run.append(dist_matrix_s)

    template_digit_labels_run = np.array(template_digit_labels_run)
    n_templates_run = len(matrix_temp_list_run)
    print(f"  Using {n_templates_run} templates in this run, labels:", template_digit_labels_run)

    # ------------------------------------------------------------------
    # Classify all samples by GW 1-NN to these templates
    # ------------------------------------------------------------------
    gw_pred_labels_run = []

    for i in range(n_samples):
        # Extract probability measure (third column) and filter valid points
        p_s = Data_selected[i, :, 2]
        valid_indices = np.where(p_s != -1)[0]
        p_s = p_s[valid_indices]
        p_s = p_s / float(p_s.sum())  # normalize

        # Extract and normalize spatial coordinates for sample i
        C_s = Data_selected[i, valid_indices, :2]
        C_s = utils.normalize_2Dpointcloud_coordinates(C_s)
        dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)

        # Compute GW distance to each template
        dists = []
        for B_temp, b_temp in zip(matrix_temp_list_run, measure_temp_list_run):
            gw_cost = ot.gromov.gromov_wasserstein2(
                dist_matrix_s,
                B_temp,
                p_s,
                b_temp,
                loss_fun="square_loss"
            )
            dists.append(gw_cost)

        dists = np.array(dists)
        nearest_template_idx = np.argmin(dists)
        pred_label = template_digit_labels_run[nearest_template_idx]
        gw_pred_labels_run.append(pred_label)

        if i % 200 == 0:
            print(f"  Processed {i} / {n_samples} samples for GW 1-NN in this run...")

    gw_pred_labels_run = np.array(gw_pred_labels_run)

    # Ground-truth labels for these samples:
    true_labels = label_selected  # same order as Data_selected

    # Evaluate
    gw_acc = accuracy_score(true_labels, gw_pred_labels_run)
    gw_cm = confusion_matrix(true_labels, gw_pred_labels_run, labels=sorted(np.unique(true_labels)))

    all_gw_accuracies.append(gw_acc)

    print(f"\n  1-NN GW template-based accuracy (run {run+1}): {gw_acc:.4f}")
    print("  Confusion matrix (true vs GW 1-NN template label):")
    print(gw_cm)
    print("\n  Classification report:")
    print(classification_report(true_labels, gw_pred_labels_run, digits=4))

time_gw1nn = time.perf_counter() - t0_gw1nn

# ----------------------------------------------------------------------
# Aggregate statistics over all GW 1-NN runs
# ----------------------------------------------------------------------
all_gw_accuracies = np.array(all_gw_accuracies)

print("\n====================================================")
print("Summary over all GW 1-NN runs (direct template-based classification)")
print("Per-run GW 1-NN accuracies:")
print(all_gw_accuracies)

print(f"\nMean GW 1-NN accuracy over {n_repeats_gw} runs: {all_gw_accuracies.mean():.4f}")
print(f"Std of GW 1-NN accuracy over {n_repeats_gw} runs: {all_gw_accuracies.std():.4f}")
print(f"Total time for GW 1-NN baseline (all runs): {time_gw1nn:.4f} s")

###################################################################################################
## (4) GW DICTIONARY LEARNING BASELINE + k-NN ON UNMIXING WEIGHTS #################################
###################################################################################################
print("\n=== GW dictionary learning baseline (on selected MNIST point clouds) ===")

from ot.gromov import (
    gromov_wasserstein_dictionary_learning,
    gromov_wasserstein_linear_unmixing,
)

# -------------------------------------------------------------------
# 1) Build (Cs, ps) list from Data_selected (same as before, but for all)
# -------------------------------------------------------------------
dist_matrices = []   # list of distance matrices
measures = []   # list of probability vectors

n_samples = Data_selected.shape[0]

for i in range(n_samples):
    # Extract probability measure (third column) and filter valid points
    p_s = Data_selected[i, :, 2]
    valid_indices = np.where(p_s != -1)[0]
    p_s = p_s[valid_indices]
    p_s = p_s / float(p_s.sum())   # normalize

    # Extract and normalize spatial coordinates (first two columns)
    C_s_points = Data_selected[i, valid_indices, :2]
    C_s_points = utils.normalize_2Dpointcloud_coordinates(C_s_points)

    # Dissimilarity matrix for this sample
    C_s = sp.spatial.distance.cdist(C_s_points, C_s_points)

    dist_matrices.append(C_s)
    measures.append(p_s)

print(f"Built {len(dist_matrices)} distance matrices for dictionary learning.")

# -------------------------------------------------------------------
# 2–4) OGDL evaluation (outer split -> learn dict on TRAIN only,
#      unmix TRAIN/TEST, inner CV for k on TRAIN, final test acc on TEST)
# -------------------------------------------------------------------
print("\n=== OGDL baseline: dict learn on TRAIN only + k-NN on unmixings ===")

labels_all = label_selected.astype(int)
idx_all = np.arange(len(dist_matrices))

candidate_ks = [1, 3, 5, 7, 9]
n_repeats_dict = 10
train_fraction = 0.7

# Dictionary hyperparams
D_dict = 2
nt = 20
q = ot.unif(nt)
reg = 0.0

# POT solver hyperparams
epochs = 5
batch_size = 16
learning_rate = 0.1
projection = "nonnegative_symmetric"

tol_outer = 1e-3
tol_inner = 1e-3
max_iter_outer = 10
max_iter_inner = 80

all_test_acc_dict = []
all_best_ks_dict = []

# Optional: record timing per run
dict_times = []
unmix_times = []
knn_times = []

def unmix_subset(Cs_subset, ps_subset, Cdict_GW, q,
                reg=0.0,
                tol_outer=1e-5, tol_inner=1e-5,
                max_iter_outer=30, max_iter_inner=300):
    W = []
    rec = []
    for C_s, p_s in zip(Cs_subset, ps_subset):
        w_s, C_emb, T_s, rec_err = gromov_wasserstein_linear_unmixing(
            C_s,
            Cdict_GW,
            p=p_s,
            q=q,
            reg=reg,
            tol_outer=tol_outer,
            tol_inner=tol_inner,
            max_iter_outer=max_iter_outer,
            max_iter_inner=max_iter_inner,
        )
        W.append(w_s)
        rec.append(rec_err)
    return np.asarray(W), np.asarray(rec)

t0_total = time.perf_counter()

for run in range(n_repeats_dict):
    print(f"\n=== OGDL outer run {run+1}/{n_repeats_dict} ===")

    train_idx, test_idx = train_test_split(
        idx_all,
        train_size=train_fraction,
        random_state=run,
        stratify=labels_all,
    )

    # Build train/test lists
    Cs_train = [dist_matrices[i] for i in train_idx]
    ps_train = [measures[i] for i in train_idx]
    y_train  = labels_all[train_idx]

    Cs_test  = [dist_matrices[i] for i in test_idx]
    ps_test  = [measures[i] for i in test_idx]
    y_test   = labels_all[test_idx]

    # --------------------------
    # (2) Learn dict on TRAIN only
    # --------------------------
    print("  Learning GW dictionary on TRAIN only...")
    t0_dict = time.perf_counter()

    Cdict_GW, log_dict = gromov_wasserstein_dictionary_learning(
        Cs=Cs_train,
        D=D_dict,
        nt=nt,
        ps=ps_train,
        q=q,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        reg=reg,
        projection=projection,
        tol_outer=tol_outer,
        tol_inner=tol_inner,
        max_iter_outer=max_iter_outer,
        max_iter_inner=max_iter_inner,
        use_log=True,
        use_adam_optimizer=True,
        verbose=True,
    )

    time_dict = time.perf_counter() - t0_dict
    dict_times.append(time_dict)
    print(f"  Dict learned. Time = {time_dict:.4f} s  | atoms shape = {Cdict_GW.shape}")

    # --------------------------
    # (3) Unmix TRAIN and TEST using this dict
    # --------------------------
    print("  Computing unmixings for TRAIN and TEST...")
    t0_unmix = time.perf_counter()

    W_train, rec_train = unmix_subset(
        Cs_train, ps_train, Cdict_GW, q,
        reg=reg, tol_outer=tol_outer, tol_inner=tol_inner,
        max_iter_outer=max_iter_outer, max_iter_inner=max_iter_inner
    )
    W_test, rec_test = unmix_subset(
        Cs_test, ps_test, Cdict_GW, q,
        reg=reg, tol_outer=tol_outer, tol_inner=tol_inner,
        max_iter_outer=max_iter_outer, max_iter_inner=max_iter_inner
    )

    time_unmix = time.perf_counter() - t0_unmix
    unmix_times.append(time_unmix)

    print(f"  Unmix done. Time = {time_unmix:.4f} s")
    print(f"  Avg recon error: train = {rec_train.mean():.6g}, test = {rec_test.mean():.6g}")
    print("  W_train shape:", W_train.shape, "| W_test shape:", W_test.shape)

    # --------------------------
    # (4) Inner CV on TRAIN to pick k; then test on TEST
    # --------------------------
    print("  Inner CV on TRAIN embeddings to choose k...")
    t0_knn = time.perf_counter()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
    mean_cv_scores = []
    for k in candidate_ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, W_train, y_train, cv=cv)
        mean_score = scores.mean()
        mean_cv_scores.append(mean_score)
        print(f"    k={k}: inner CV acc = {mean_score:.4f}")

    best_k = candidate_ks[int(np.argmax(mean_cv_scores))]
    all_best_ks_dict.append(best_k)
    print(f"  -> Best k = {best_k}")

    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(W_train, y_train)
    y_pred = knn_best.predict(W_test)

    test_acc = accuracy_score(y_test, y_pred)
    all_test_acc_dict.append(test_acc)

    time_knn = time.perf_counter() - t0_knn
    knn_times.append(time_knn)

    print(f"  Test accuracy = {test_acc:.4f}")
    print("  Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=np.unique(labels_all)))
    print(classification_report(y_test, y_pred, digits=4))

time_total = time.perf_counter() - t0_total

time_dict_learn = float(np.sum(dict_times))
time_unmix      = float(np.sum(unmix_times))
time_knn_dict   = float(np.sum(knn_times))


all_test_acc_dict = np.asarray(all_test_acc_dict)
all_best_ks_dict = np.asarray(all_best_ks_dict)

print("\n====================================================")
print("OGDL baseline: summary over all runs")
print("Per-run test accuracies:", all_test_acc_dict)
print(f"Mean ± std: {all_test_acc_dict.mean():.4f} ± {all_test_acc_dict.std():.4f}")
print("Best-k per run:", all_best_ks_dict)

print("\nTiming per run (seconds):")
print("Dict learn:", np.asarray(dict_times))
print("Unmix:", np.asarray(unmix_times))
print("k-NN (CV+fit+test):", np.asarray(knn_times))
print(f"Total time (all runs): {time_total:.4f} s")


###################################################################################################
## FINAL SUMMARY: ACCURACIES + RUNTIMES ###########################################################
###################################################################################################
print("\n\n================ FINAL SUMMARY (methods, accuracies, runtimes) ================")

# 1) λ-space method: fixed-point
lambda_fp_mean_acc = all_test_accuracies_fp.mean()
lambda_fp_std_acc = all_test_accuracies_fp.std()
lambda_fp_total_time = time_lambdas_fp + time_knn_fp

print(f"λ-space GW (fixed-point) + k-NN: mean acc = {lambda_fp_mean_acc:.4f}, "
      f"std = {lambda_fp_std_acc:.4f}, "
      f"feature time = {time_lambdas_fp:.4f} s, k-NN time = {time_knn_fp:.4f} s, "
      f"total ≈ {lambda_fp_total_time:.4f} s")

# 2) λ-space method: blow-up (gradient)
lambda_bu_mean_acc = all_test_accuracies_bu.mean()
lambda_bu_std_acc = all_test_accuracies_bu.std()
lambda_bu_total_time = time_lambdas_bu + time_knn_bu

print(f"λ-space GW (blow-up) + k-NN: mean acc = {lambda_bu_mean_acc:.4f}, "
      f"std = {lambda_bu_std_acc:.4f}, "
      f"feature time = {time_lambdas_bu:.4f} s, k-NN time = {time_knn_bu:.4f} s, "
      f"total ≈ {lambda_bu_total_time:.4f} s")

# 3) GW 1-NN baseline
gw1nn_mean_acc = all_gw_accuracies.mean()
gw1nn_std_acc = all_gw_accuracies.std()
gw1nn_total_time = time_gw1nn

print(f"GW 1-NN (templates): mean acc = {gw1nn_mean_acc:.4f}, std = {gw1nn_std_acc:.4f}, "
      f"total time (all runs) = {gw1nn_total_time:.4f} s")

# 4) GW dictionary learning baseline
dict_mean_acc = all_test_acc_dict.mean()
dict_std_acc = all_test_acc_dict.std()
dict_total_time = time_dict_learn + time_unmix + time_knn_dict

print(f"GW dictionary + k-NN: mean acc = {dict_mean_acc:.4f}, std = {dict_std_acc:.4f}, "
      f"dict learn = {time_dict_learn:.4f} s, unmix = {time_unmix:.4f} s, "
      f"k-NN time = {time_knn_dict:.4f} s, total ≈ {dict_total_time:.4f} s")

# Optional: LaTeX table
print("\nLaTeX table suggestion:\n")
print(r"""
\begin{table}[h]
\centering
\begin{tabular}{lccc}
\toprule
Method & Accuracy (mean $\pm$ std) & Runtime (s) \\\midrule
GW-barycentric (fixed-point) + k-NN 
  & %.3f $\pm$ %.3f & %.2f \\\
GW-barycentric (blow-up) + k-NN 
  & %.3f $\pm$ %.3f & %.2f \\\
GW 1-NN (templates) 
  & %.3f $\pm$ %.3f & %.2f \\\
GW dictionary + k-NN 
  & %.3f $\pm$ %.3f & %.2f \\\bottomrule
\end{tabular}
\caption{Performance and runtime comparison of GW-based classification methods on the 2D point cloud MNIST subset ($|\mathcal{D}|=%d$).}
\label{tab:gw_mnist_comparison}
\end{table}
""" % (
    lambda_fp_mean_acc, lambda_fp_std_acc, lambda_fp_total_time,
    lambda_bu_mean_acc, lambda_bu_std_acc, lambda_bu_total_time,
    gw1nn_mean_acc, gw1nn_std_acc, gw1nn_total_time,
    dict_mean_acc, dict_std_acc, dict_total_time,
    Data_selected.shape[0],
))
