## This notebook uses the 2D point cloud MNIST dataset.
##
## We compare:
##  (1) GW-barycentric λ (fixed-point) + k-NN
##  (2) GW-barycentric λ (blow-up / gradient) + k-NN
##  (3) GW 1-NN (templates, direct GW)
##  (4) GW dictionary learning + k-NN on unmixing weights
##
## KNN is applied in the GW coordinate / feature space

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
    adjusted_rand_score, #<--- K-Means
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


# def repeated_cv_knn(features, labels, candidate_ks, n_repeats=20, train_fraction=0.7):
#     """
#     Run repeated train/test splits with inner CV to select k in k-NN.
#
#     Parameters
#     ----------
#     features : ndarray (n_samples, d)
#         Feature matrix.
#     labels : ndarray (n_samples,)
#         Class labels.
#     candidate_ks : list[int]
#         Candidate k values for k-NN.
#     n_repeats : int
#         Number of outer repetitions.
#     train_fraction : float
#         Fraction of data used for training in each outer split.
#
#     Returns
#     -------
#     all_test_acc : np.ndarray of shape (n_repeats,)
#         Test accuracies for each outer run.
#     all_best_ks : np.ndarray of shape (n_repeats,)
#         Chosen k for each outer run.
#     total_time : float
#         Wall-clock time for the whole procedure.
#     """
#     all_test_acc = []
#     all_best_ks = []
#
#     t0 = time.perf_counter()
#
#     for run in range(n_repeats):
#         print(f"\n=== Outer run {run+1}/{n_repeats} ===")
#
#         X_train, X_test, y_train, y_test = train_test_split(
#             features,
#             labels,
#             train_size=train_fraction,
#             random_state=run,
#             stratify=labels,
#         )
#
#         print("  Train size:", X_train.shape[0])
#         print("  Test size:", X_test.shape[0])
#
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
#
#         mean_cv_scores = []
#         for k in candidate_ks:
#             knn = KNeighborsClassifier(n_neighbors=k)
#             scores = cross_val_score(knn, X_train, y_train, cv=cv)
#             mean_score = scores.mean()
#             mean_cv_scores.append(mean_score)
#             print(f"  k = {k}: CV accuracy = {mean_score:.4f}")
#
#         best_idx = int(np.argmax(mean_cv_scores))
#         best_k = candidate_ks[best_idx]
#         all_best_ks.append(best_k)
#         print(f"  -> Best k on this run = {best_k}")
#
#         knn_best = KNeighborsClassifier(n_neighbors=best_k)
#         knn_best.fit(X_train, y_train)
#         y_pred = knn_best.predict(X_test)
#
#         test_acc = accuracy_score(y_test, y_pred)
#         all_test_acc.append(test_acc)
#
#         print(f"  Test accuracy (best k={best_k}): {test_acc:.4f}")
#         print("  Confusion matrix (true vs predicted):")
#         print(confusion_matrix(y_test, y_pred, labels=np.unique(labels)))
#         print("\n  Classification report:")
#         print(classification_report(y_test, y_pred, digits=4))
#
#     total_time = time.perf_counter() - t0
#
#     return np.array(all_test_acc), np.array(all_best_ks), total_time


def repeated_cv_knn(features, labels, candidate_ks, n_repeats=20, train_fraction=0.7, precomputed=False):
    """
    Same as your repeated_cv_knn, but if precomputed=True then
    'features' must be a full (n_samples, n_samples) distance matrix,
    and we run k-NN with metric='precomputed' with correct slicing.
    """
    all_test_acc = []
    all_best_ks = []

    t0 = time.perf_counter()
    n = len(labels)
    idx_all = np.arange(n)

    for run in range(n_repeats):
        print(f"\n=== Outer run {run+1}/{n_repeats} ===")

        idx_train, idx_test, y_train, y_test = train_test_split(
            idx_all,
            labels,
            train_size=train_fraction,
            random_state=run,
            stratify=labels,
        )

        print("  Train size:", len(idx_train))
        print("  Test size:", len(idx_test))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)

        mean_cv_scores = []

        for k in candidate_ks:
            fold_scores = []

            for fold_tr_pos, fold_va_pos in cv.split(np.zeros(len(idx_train)), y_train):
                # indices in the ORIGINAL dataset
                fold_tr_idx = idx_train[fold_tr_pos]
                fold_va_idx = idx_train[fold_va_pos]

                k_eff = min(k, len(fold_tr_idx))

                if not precomputed:
                    X_tr = features[fold_tr_idx]
                    X_va = features[fold_va_idx]
                    knn = KNeighborsClassifier(n_neighbors=k_eff)
                    knn.fit(X_tr, labels[fold_tr_idx])
                    y_pred = knn.predict(X_va)
                else:
                    # Correct slicing for precomputed distances
                    D_tr = features[np.ix_(fold_tr_idx, fold_tr_idx)]   # square
                    D_va = features[np.ix_(fold_va_idx, fold_tr_idx)]   # val-to-train
                    knn = KNeighborsClassifier(n_neighbors=k_eff, metric="precomputed")
                    knn.fit(D_tr, labels[fold_tr_idx])
                    y_pred = knn.predict(D_va)

                fold_scores.append(accuracy_score(labels[fold_va_idx], y_pred))

            mean_score = float(np.mean(fold_scores))
            mean_cv_scores.append(mean_score)
            print(f"  k = {k}: CV accuracy = {mean_score:.4f}")

        best_idx = int(np.argmax(mean_cv_scores))
        best_k = candidate_ks[best_idx]
        all_best_ks.append(best_k)
        print(f"  -> Best k on this run = {best_k}")

        # Fit on full TRAIN, test on TEST
        if not precomputed:
            X_train = features[idx_train]
            X_test  = features[idx_test]
            knn_best = KNeighborsClassifier(n_neighbors=best_k)
            knn_best.fit(X_train, labels[idx_train])
            y_pred = knn_best.predict(X_test)
        else:
            D_train = features[np.ix_(idx_train, idx_train)]
            D_test  = features[np.ix_(idx_test, idx_train)]
            k_eff = min(best_k, len(idx_train))
            knn_best = KNeighborsClassifier(n_neighbors=k_eff, metric="precomputed")
            knn_best.fit(D_train, labels[idx_train])
            y_pred = knn_best.predict(D_test)

        test_acc = accuracy_score(labels[idx_test], y_pred)
        all_test_acc.append(test_acc)

        print(f"  Test accuracy (best k={best_k}): {test_acc:.4f}")
        print("  Confusion matrix (true vs predicted):")
        print(confusion_matrix(labels[idx_test], y_pred, labels=np.unique(labels)))
        print("\n  Classification report:")
        print(classification_report(labels[idx_test], y_pred, digits=4))

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
n_per_class = 50
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

        # Center and normalize coordinates to fit within the unit square [0,1]²
        C_s = utils.normalize_2Dpointcloud_coordinates(C_s)

        # Compute the pairwise Euclidean distance matrix for C_s
        dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)
        matrix_temp_list.append(dist_matrix_s)

print("Random templates, extracted")

# ## PLOT TEMPLATES (Optional)  #####################################################################
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


## Compute all distance matrices and respectively node-measures for all samples in the data set
n_samples = Data_selected.shape[0]

measures = []   # list of template measures
matrices = []    # list of template dissimilarity matrices

for i in range(n_samples):
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
    C_s = sp.spatial.distance.cdist(C_s, C_s)

    matrices.append(C_s)
    measures.append(p_s)





###################################################################################################
## COMPUTE GW-BARYCENTRIC COORDINATES (lambdas) FOR ALL SELECTED SAMPLES ##########################
##  -> we compute BOTH: fixed-point (FP) and gradient via blow-up (BU) ############################
###################################################################################################
print("Compute GW-barycentric coordinates for all selected samples from data set")
print("We will both methods: (1) fixed-point and (2) blow-up.\n")

all_lambda_fp_list = []    # [label | λ_fp]
all_lambda_bu_list = []    # [label | λ_bu]

average_bu_size = 0.0

# --- fixed-point method -------------------------------------------------------------------------
print(">>> Computing λ via fixed-point method (constraints) for all samples...")
t0_lambdas_fp = time.perf_counter()

for i in range(n_samples):
    dist_matrix_s = matrices[i]
    p_s = measures[i]

    # Fixed-point λ
    _, lambdas_fp = utils.get_lambdas_constraints(
        matrix_temp_list, measure_temp_list, dist_matrix_s, p_s
    )

    sample_label = label_selected[i]
    all_lambda_fp_list.append(np.concatenate(([sample_label], lambdas_fp)))

    if i % 50 == 0:
        print(f"  [FP] Processed {i} samples...")

time_lambdas_fp = time.perf_counter() - t0_lambdas_fp
print(f"Time to compute all GW-barycentric coordinates (fixed-point): {time_lambdas_fp:.4f} s")

## Uncomment to test BU
# # --- gradient via blow-up method ----------------------------------------------------------------
# print("\n>>> Computing λ via gradient method (blow-up) for all samples...")
# t0_lambdas_bu = time.perf_counter()
#
# for i in range(n_samples):
#     dist_matrix_s = matrices[i]
#     p_s = measures[i]
#
#     # Blow-up + gradient λ
#     dist_matrix_s_bu, p_s_bu, temp_blow_up = utils.blow_up(
#         matrix_temp_list, measure_temp_list, dist_matrix_s, p_s
#     )
#     _, lambdas_bu = utils.get_lambdas_blowup(temp_blow_up, dist_matrix_s_bu, p_s_bu)
#
#     average_bu_size += dist_matrix_s_bu.shape[0]
#
#     sample_label = label_selected[i]
#     all_lambda_bu_list.append(np.concatenate(([sample_label], lambdas_bu)))
#
#     if i % 50 == 0:
#         print(f"  [BU] Processed {i} samples...")
#
# time_lambdas_bu = time.perf_counter() - t0_lambdas_bu
# average_bu_size /= Data_selected.shape[0]
#
# print(f"Average blow-up size: {average_bu_size}")
# print(f"Time to compute all GW-barycentric coordinates (blow-up): {time_lambdas_bu:.4f} s")

print("Barycentric coordinates for all samples, computed (both methods).")



# Convert to arrays
all_lambda_fp = np.array(all_lambda_fp_list)

# #Uncomment to test BU
# all_lambda_bu = np.array(all_lambda_bu_list)


# Labels are the same for both methods
all_labels = all_lambda_fp[:, 0].astype(int)

# Features (λ) for each method
all_features_fp = all_lambda_fp[:, 1:]

# #Uncomment to test BU
# all_features_bu = all_lambda_bu[:, 1:]



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

## Uncomment to test BU
# # --- Blow-up λ ----------------------------------------------------------------------------------
# print("\n>>> λ (blow-up) + k-NN")
# all_test_accuracies_bu, all_best_ks_bu, time_knn_bu = repeated_cv_knn(
#     all_features_bu, all_labels, candidate_ks, n_repeats=n_repeats, train_fraction=train_fraction
# )
#
# print("\n====================================================")
# print("Blow-up λ: summary over all runs")
# print("Per-run test accuracies:")
# print(all_test_accuracies_bu)
#
# print(f"\nMean test accuracy (BU) over {n_repeats} runs: {all_test_accuracies_bu.mean():.4f}")
# print(f"Std of test accuracy (BU) over {n_repeats} runs: {all_test_accuracies_bu.std():.4f}")
# print(f"Total time for BU λ-method k-NN (CV + repeats): {time_knn_bu:.4f} s")
#
# unique_ks_bu, counts_ks_bu = np.unique(all_best_ks_bu, return_counts=True)
# print("\nBest-k frequency over runs (BU):")
# for k_val, count in zip(unique_ks_bu, counts_ks_bu):
#     print(f"  k = {k_val}: chosen {count} times out of {n_repeats}")




###################################################################################################
## ORIGINAL GW-EMBEDDING to S templates + k-NN (same protocol as FP/BU)
##     Yi -> ( GW(X1,Yi), GW(X2,Yi), ..., GW(XS,Yi) )
###################################################################################################
print("\n(3) Naive GW embedding to S fixed templates + k-NN (same protocol as FP/BU)")

# -------------------------------------------------------------------
# Templates: S = n_classes * n_temp total templates
# -------------------------------------------------------------------
S = n_classes * n_temp
assert len(matrix_temp_list) == S, f"Expected {S} templates, got {len(matrix_temp_list)}"
assert len(measure_temp_list) == S, f"Expected {S} template measures, got {len(measure_temp_list)}"

# -------------------------------------------------------------------
# Build embedding: gw_embed[i, j] = GW(Yi, Xj)
# where Xj is the j-th template (distance matrix + measure).
# -------------------------------------------------------------------
gw_embed = np.zeros((n_samples, S), dtype=float)

t0_embed = time.perf_counter()

for i in range(n_samples):
    C_s = matrices[i]
    p_s = measures[i]

    # compute GW to each template
    for j in range(S):
        B_j = matrix_temp_list[j]
        b_j = measure_temp_list[j]

        d = ot.gromov.gromov_wasserstein2(
            C_s, B_j, p_s, b_j, loss_fun="square_loss"
        )

        # Option A (default): store GW^2 cost
        gw_embed[i, j] = float(d)

        # Option B (distance-like scaling): uncomment if sqrt(cost) is preferred
        # gw_embed[i, j] = float(np.sqrt(d))

    if i % 50 == 0:
        print(f"  Built embedding for {i}/{n_samples} samples...")

time_embed = time.perf_counter() - t0_embed
print(f"Time to build {S}D GW embedding: {time_embed:.4f} s")
print("GW embedding shape:", gw_embed.shape)

# -------------------------------------------------------------------
# k-NN on GW embedding using YOUR SAME FP/BU evaluation routine
# (Repeated train/test splits + inner 5-fold CV to pick k)
# -------------------------------------------------------------------
candidate_ks = [1, 3, 5, 7, 9]
n_repeats_gw_naive = n_repeats
train_fraction = 0.7  # keep consistent with FP/BU (or change as desired)

print("\n>>> GW-embedding (S-D) + k-NN (your repeated_cv_knn protocol)")
all_test_acc_gwemb, all_best_ks_gwemb, time_knn_gwemb = repeated_cv_knn(
    gw_embed, label_selected.astype(int),
    candidate_ks,
    n_repeats=n_repeats_gw_naive,
    train_fraction=train_fraction
)



print("\n====================================================")
print("GW-embedding: summary over all runs")
print("Per-run test accuracies:")
print(all_test_acc_gwemb)

print(f"\nMean test accuracy over {n_repeats} runs: {all_test_acc_gwemb.mean():.4f}")
print(f"Std  test accuracy over {n_repeats} runs: {all_test_acc_gwemb.std():.4f}")
print(f"Embedding time = {time_embed:.4f} s")
print(f"k-NN time (CV + repeats) = {time_knn_gwemb:.4f} s")
print(f"Total ≈ {(time_embed + time_knn_gwemb):.4f} s")

unique_ks, counts_ks = np.unique(all_best_ks_gwemb, return_counts=True)
print("\nBest-k frequency over runs (GW-embedding):")
for k_val, count in zip(unique_ks, counts_ks):
    print(f"  k = {k_val}: chosen {count} times out of {n_repeats}")
###################################################################################################



###################################################################################################
## (3b) GW k-NN among samples (precomputed GW distances) + inner 5-fold CV for k
##      Uses the NEW repeated_cv_knn(..., precomputed=True)
###################################################################################################
print("\n(3b) GW k-NN among samples (precomputed GW distances)")

labels = label_selected.astype(int)
n_samples = len(matrices)
assert len(measures) == n_samples, "matrices and measures must have the same length"

# -------------------------------------------------------------------
# 1) Precompute all pairwise GW distances (symmetric matrix)
# -------------------------------------------------------------------
print("\nPrecomputing pairwise GW distance matrix between ALL samples...")
D_gw = np.zeros((n_samples, n_samples), dtype=float)

t0_distmat = time.perf_counter()

for i in range(n_samples):
    C_i = matrices[i]
    p_i = measures[i]
    for j in range(i + 1, n_samples):
        C_j = matrices[j]
        p_j = measures[j]

        d_ij = ot.gromov.gromov_wasserstein2(
            C_i, C_j, p_i, p_j, loss_fun="square_loss"
        )

        D_gw[i, j] = float(d_ij)
        D_gw[j, i] = float(d_ij)

    if i % 50 == 0:
        print(f"{i}/{n_samples} samples...")

time_distmat = time.perf_counter() - t0_distmat
print(f"Done. Time for pairwise GW distance matrix: {time_distmat:.4f} s")
print("D_gw shape:", D_gw.shape)

# -------------------------------------------------------------------
# 2) Run repeated outer splits + inner 5-fold CV to pick k,
#    but using PRECOMPUTED distances
# -------------------------------------------------------------------
print("\nRepeated GW k-NN classification using precomputed distances (choose k by 5-fold CV)")

candidate_ks = [1, 3, 5, 7, 9]
n_repeats_gw_sample = n_repeats          # keep consistent
train_fraction_gw_sample = train_fraction

all_test_acc_gw_sample, all_best_ks_gw_sample, time_knn_gw_sample = repeated_cv_knn(
    D_gw,
    labels,
    candidate_ks,
    n_repeats=n_repeats_gw_sample,
    train_fraction=train_fraction_gw_sample,
    precomputed=True
)

print("\n====================================================")
print("Summary: GW k-NN among samples (precomputed GW distances)")
print("Per-run test accuracies:")
print(all_test_acc_gw_sample)

print(f"\nMean test accuracy over {n_repeats_gw_sample} runs: {all_test_acc_gw_sample.mean():.4f}")
print(f"Std  test accuracy over {n_repeats_gw_sample} runs: {all_test_acc_gw_sample.std():.4f}")
print(f"Time for distance matrix: {time_distmat:.4f} s")
print(f"Time for repeated eval (CV+fit+test): {time_knn_gw_sample:.4f} s")
print(f"Total ≈ {(time_distmat + time_knn_gw_sample):.4f} s")
print("\nBest-k frequency (GW k-NN among samples):")
unique_ks, counts_ks = np.unique(all_best_ks_gw_sample, return_counts=True)
for k_val, count in zip(unique_ks, counts_ks):
    print(f"  k = {k_val}: chosen {count} times out of {n_repeats_gw_sample}")

###################################################################################################





###################################################################################################
## GW DICTIONARY LEARNING BASELINE + k-NN ON UNMIXING WEIGHTS ####################################
###################################################################################################
print("\n=== GW dictionary learning baseline (on selected MNIST point clouds) ===")

from ot.gromov import (
    gromov_wasserstein_dictionary_learning,
    gromov_wasserstein_linear_unmixing,
)

Cs_list = matrices
ps_list = measures

# -------------------------------------------------------------------
# Learn GW dictionary on these Cs_list
# -------------------------------------------------------------------
D_dict = n_classes * n_temp         # number of atoms in the dictionary
nt = 50            # number of nodes per atom
q = ot.unif(nt)    # uniform reference measure on atoms
reg = 0.0          # regularization

print("\nLearning GW dictionary...")

t0_dict_learn = time.perf_counter()

Cdict_GW, log_dict = gromov_wasserstein_dictionary_learning(
    Cs=Cs_list,
    D=D_dict,
    nt=nt,
    ps=ps_list,
    q=q,
    epochs=5,           #<---- to change
    batch_size=16,
    learning_rate=0.1,
    reg=reg,
    projection="nonnegative_symmetric",
    tol_outer=1e-5,
    tol_inner=1e-5,
    max_iter_outer=30,
    max_iter_inner=300,
    use_log=True,
    use_adam_optimizer=True,
    verbose=True,
)

time_dict_learn = time.perf_counter() - t0_dict_learn

print("GW dictionary learned. Shape of atoms array:", Cdict_GW.shape)
print(f"Time for GW dictionary learning: {time_dict_learn:.4f} s")

# -------------------------------------------------------------------
# Compute unmixing weights w_s for each sample (GW linear unmixing)
# -------------------------------------------------------------------
print("\nComputing unmixing weights (embeddings) for each sample...")

unmixings = []
recon_errors = []

t0_unmix = time.perf_counter()

for C_s, p_s in zip(Cs_list, ps_list):
    w_s, C_emb, T_s, rec_err = gromov_wasserstein_linear_unmixing(
        C_s,
        Cdict_GW,
        p=p_s,
        q=q,
        reg=reg,
        tol_outer=1e-5,
        tol_inner=1e-5,
        max_iter_outer=30,
        max_iter_inner=300,
    )
    unmixings.append(w_s)
    recon_errors.append(rec_err)

time_unmix = time.perf_counter() - t0_unmix

unmixings = np.array(unmixings)
recon_errors = np.array(recon_errors)

print("Unmixings shape:", unmixings.shape)
print("Average reconstruction error:", recon_errors.mean())
print(f"Time for GW linear unmixing (all samples): {time_unmix:.4f} s")

# -------------------------------------------------------------------
#    Use unmixing weights as features for classification with k-NN
#    Cross-validated k selection + repeated train/test splits
# -------------------------------------------------------------------
print("\nCross-validated k-NN on GW-dictionary unmixing features")

dict_features = unmixings
dict_labels = label_selected.astype(int)

candidate_ks = [1, 3, 5, 7, 9]
n_repeats_dict = n_repeats
train_fraction = 0.7

t0_knn_dict = time.perf_counter()

all_test_acc_dict, all_best_ks_dict, time_knn_dict_inner = repeated_cv_knn(
    dict_features, dict_labels, candidate_ks,
    n_repeats=n_repeats_dict, train_fraction=train_fraction
)

time_knn_dict = time.perf_counter() - t0_knn_dict

# Aggregate stats
all_test_acc_dict = np.array(all_test_acc_dict)
all_best_ks_dict = np.array(all_best_ks_dict)

print("\n====================================================")
print("GW-dictionary baseline: summary over all runs")
print("Per-run test accuracies:")
print(all_test_acc_dict)

print(f"\nMean test accuracy over {n_repeats_dict} runs: {all_test_acc_dict.mean():.4f}")
print(f"Std of test accuracy over {n_repeats_dict} runs: {all_test_acc_dict.std():.4f}")
print(f"Total time for GW-dict k-NN (CV + repeats): {time_knn_dict:.4f} s")

unique_ks, counts_ks = np.unique(all_best_ks_dict, return_counts=True)
print("\nBest-k frequency over runs (GW-dict baseline):")
for k_val, count in zip(unique_ks, counts_ks):
    print(f"  k = {k_val}: chosen {count} times out of {n_repeats_dict}")

###################################################################################################
## FINAL SUMMARY: ACCURACIES + RUNTIMES ###########################################################
###################################################################################################
print("\n\n================ FINAL SUMMARY (methods, accuracies, runtimes) ================")

# 1) λ-space method: fixed-point
lambda_fp_mean_acc = all_test_accuracies_fp.mean()
lambda_fp_std_acc = all_test_accuracies_fp.std()
lambda_fp_feature_time = time_lambdas_fp
lambda_fp_knn_time = time_knn_fp
lambda_fp_total_time = lambda_fp_feature_time + lambda_fp_knn_time

print(f"λ-space GW (fixed-point) + k-NN: mean acc = {lambda_fp_mean_acc:.4f}, "
      f"std = {lambda_fp_std_acc:.4f}, "
      f"feature time = {lambda_fp_feature_time:.4f} s, k-NN time = {lambda_fp_knn_time:.4f} s, "
      f"total ≈ {lambda_fp_total_time:.4f} s")


## Uncomment to test BU
# # 2) λ-space method: blow-up (gradient)
# lambda_bu_mean_acc = all_test_accuracies_bu.mean()
# lambda_bu_std_acc = all_test_accuracies_bu.std()
# lambda_bu_feature_time = time_lambdas_bu
# lambda_bu_knn_time = time_knn_bu
# lambda_bu_total_time = lambda_bu_feature_time + lambda_bu_knn_time

# print(f"λ-space GW (blow-up) + k-NN: mean acc = {lambda_bu_mean_acc:.4f}, "
#       f"std = {lambda_bu_std_acc:.4f}, "
#       f"feature time = {lambda_bu_feature_time:.4f} s, k-NN time = {lambda_bu_knn_time:.4f} s, "
#       f"total ≈ {lambda_bu_total_time:.4f} s")


# 3) NEW baseline: GW-embedding + k-NN
gwemb_mean_acc = all_test_acc_gwemb.mean()
gwemb_std_acc = all_test_acc_gwemb.std()
gwemb_feature_time = time_embed
gwemb_knn_time = time_knn_gwemb
gwemb_total_time = gwemb_feature_time + gwemb_knn_time

print(f"GW-embedding + k-NN: mean acc = {gwemb_mean_acc:.4f}, "
      f"std = {gwemb_std_acc:.4f}, "
      f"feature time = {gwemb_feature_time:.4f} s, k-NN time = {gwemb_knn_time:.4f} s, "
      f"total ≈ {gwemb_total_time:.4f} s")


# 3b) GW k-NN among samples (precomputed sample-sample GW distances)
gwsample_mean_acc = all_test_acc_gw_sample.mean()
gwsample_std_acc  = all_test_acc_gw_sample.std()
gwsample_feature_time = time_distmat          # building the GW distance matrix
gwsample_knn_time     = time_knn_gw_sample    # repeated k-NN eval time (CV+fit+test)
gwsample_total_time   = gwsample_feature_time + gwsample_knn_time

print(f"GW k-NN among samples + (precomputed GW distances): mean acc = {gwsample_mean_acc:.4f}, "
      f"std = {gwsample_std_acc:.4f}, "
      f"feature time = {gwsample_feature_time:.4f} s, k-NN/eval time = {gwsample_knn_time:.4f} s, "
      f"total ≈ {gwsample_total_time:.4f} s")


# 4) GW dictionary learning baseline
dict_mean_acc = all_test_acc_dict.mean()
dict_std_acc = all_test_acc_dict.std()
dict_feature_time = time_dict_learn + time_unmix
dict_knn_time = time_knn_dict
dict_total_time = dict_feature_time + dict_knn_time

print(f"GW dictionary + k-NN: mean acc = {dict_mean_acc:.4f}, std = {dict_std_acc:.4f}, "
      f"feature time (learn+unmix) = {dict_feature_time:.4f} s, k-NN time = {dict_knn_time:.4f} s, "
      f"total ≈ {dict_total_time:.4f} s")

# # LaTeX table

# print(r"""
# \begin{table}[h]
# \centering
# \begin{tabular}{lccc}
# \toprule
# Method & Accuracy (mean $\pm$ std) & Runtime (s) \\\midrule
# GW-barycentric (fixed-point) + k-NN
#   & %.3f $\pm$ %.3f & %.2f \\\
# GW-barycentric (blow-up) + k-NN
#   & %.3f $\pm$ %.3f & %.2f \\\
# GW-embedding  + k-NN
#   & %.3f $\pm$ %.3f & %.2f \\\
# GW dictionary + k-NN
#   & %.3f $\pm$ %.3f & %.2f \\\bottomrule
# \end{tabular}
# \caption{Performance and runtime comparison of GW-based classification methods on the 2D point cloud MNIST subset ($%d$).}
# \label{tab:gw_mnist_comparison}
# \end{table}
# """ % (
#     lambda_fp_mean_acc, lambda_fp_std_acc, lambda_fp_total_time,
#     lambda_bu_mean_acc, lambda_bu_std_acc, lambda_bu_total_time,
#     gwemb_mean_acc, gwemb_std_acc, gwemb_total_time,
#     dict_mean_acc, dict_std_acc, dict_total_time,
#     Data_selected.shape[0],
# ))

