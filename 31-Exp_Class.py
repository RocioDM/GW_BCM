## This notebook uses the 2D point cloud MNIST dataset.
##
## We compare
#  our methods
##  (1) GW-barycentric λ (fixed-point) + k-NN ---> KNN is applied in the GW coordinate / feature space
##  (2) GW-barycentric λ (blow-up / gradient) + k-NN
##  with
##  (3) GW 1-NN (templates, direct GW)
##  (4) GW dictionary learning + k-NN on unmixing weights:
##  by C. Vincent-Cuaz, T. Vayer, R. Flamary, M. Corneli, N. Courty, Online Graph Dictionary Learning, International Conference on Machine Learning (ICML), 2021.
##

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

from ot.gromov import (
    gromov_wasserstein_dictionary_learning,
    gromov_wasserstein_linear_unmixing,
)

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


def repeated_cv_knn(features, labels, candidate_ks, n_repeats=20, train_fraction=0.7):
    """
    Run repeated train/test splits with inner cross validation (cv) to select k in k-NN.

    Parameters:
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
n_per_class = 900
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
##  -> we compute BOTH: fixed-point (FP) and gradient via blow-up (BU) ############################
###################################################################################################
print("Compute GW-barycentric coordinates for all selected samples from data set")
print("We will compute GW coordinates with both methods: (1) fixed-point and (2) blow-up.\n")

all_lambda_fp_list = []    # [label | λ_fp]
all_lambda_bu_list = []    # [label | λ_bu]

average_bu_size = 0.0

# --- fixed-point method -------------------------------------------------------------------------
print(">>> Computing λ via fixed-point method (constraints) for all samples...")
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
    # Blow-up + gradient λ
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
print("\nBarycentric coordinates for all samples, computed (both methods).")

# Convert to arrays
all_lambda_fp = np.array(all_lambda_fp_list)
all_lambda_bu = np.array(all_lambda_bu_list)

# Labels are the same for both methods
all_labels = all_lambda_fp[:, 0].astype(int)

# Features (λ) for each method
all_features_fp = all_lambda_fp[:, 1:]
all_features_bu = all_lambda_bu[:, 1:]


# For unsupervised clustering / visualization we use FP λ by default (change to bu when comparison is needed)
#all_features = all_features_fp



###################################################################################################
## CROSS-VALIDATED K-NN ON GW-BARYCENTRIC COORDINATES (BOTH METHODS) ##############################
###################################################################################################
print("\nCross-validated k-NN on GW-barycentric coordinates (fixed-point and blow-up)")

candidate_ks = [1, 3, 5, 7, 9]
n_repeats = 20
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
## 1-NN CLASSIFICATION USING DIRECT GW DISTANCE TO TEMPLATES (NO λ-COORDINATES) ###################
###################################################################################################
print("\nRepeated 1-NN classification using direct GW distance to templates (no barycentric coordinates)")

# We classify each sample in Data_selected by:
#   y_pred(i) = label of the template with smallest GW distance to sample i.
# We repeat this experiment with different randomly chosen templates to
# obtain a distribution of accuracies.

n_samples = Data_selected.shape[0]
n_repeats_gw = 5      # number of repetitions
n_temp = 1             # templates per digit



# Store per-run accuracies and per-run times
all_gw_accuracies = []
all_gw_times = []

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
            inds_d = digit_indices[digit]
            ind = inds_d[np.random.randint(len(inds_d))]
            template_digit_labels_run.append(digit)

            p_s = Data[ind, :, 2]
            valid_indices = np.where(p_s != -1)[0]
            p_s = p_s[valid_indices]
            p_s = p_s / float(p_s.sum())

            C_s = Data[ind, valid_indices, :2]
            C_s = utils.normalize_2Dpointcloud_coordinates(C_s)
            dist_matrix_s = sp.spatial.distance.cdist(C_s, C_s)

            measure_temp_list_run.append(p_s)
            matrix_temp_list_run.append(dist_matrix_s)

    template_digit_labels_run = np.array(template_digit_labels_run)

    # ------------------------------------------------------------------
    # Classify all samples by GW 1-NN to these templates
    # ------------------------------------------------------------------

    # --- timing: measure the runtime for THIS run ---
    t0 = time.perf_counter()

    gw_pred_labels_run = []

    for i in range(n_samples):

        dists = []
        for B_temp, b_temp in zip(matrix_temp_list_run, measure_temp_list_run):
            gw_cost = ot.gromov.gromov_wasserstein2(
                dist_matrices[i],
                B_temp,
                measures[i],
                b_temp,
                loss_fun="square_loss"
            )
            dists.append(gw_cost)

        pred_label = template_digit_labels_run[np.argmin(dists)]
        gw_pred_labels_run.append(pred_label)

        if i % 200 == 0:
            print(f"  Processed {i} / {n_samples} samples for GW 1-NN in this run...")

    # Record time for this run
    run_time = time.perf_counter() - t0
    all_gw_times.append(run_time)

    # Evaluate
    gw_pred_labels_run = np.array(gw_pred_labels_run)
    true_labels = label_selected

    gw_acc = accuracy_score(true_labels, gw_pred_labels_run)
    all_gw_accuracies.append(gw_acc)

    print(f"\n  1-NN GW template-based accuracy (run {run+1}): {gw_acc:.4f}")
    print(f"  Runtime for run {run+1}: {run_time:.2f} s")
    print("  Confusion matrix (true vs predicted):")
    print(confusion_matrix(true_labels, gw_pred_labels_run))
    print("\n  Classification report:")
    print(classification_report(true_labels, gw_pred_labels_run, digits=4))


# ----------------------------------------------------------------------
# Aggregate statistics over all runs
# ----------------------------------------------------------------------

all_gw_accuracies = np.array(all_gw_accuracies)
all_gw_times = np.array(all_gw_times)

print("\n====================================================")
print("Summary over all GW 1-NN runs (direct template-based classification)")
print("Per-run GW 1-NN accuracies:")
print(all_gw_accuracies)

print(f"\nMean GW 1-NN accuracy over {n_repeats_gw} runs: {all_gw_accuracies.mean():.4f}")
print(f"Std of GW 1-NN accuracy: {all_gw_accuracies.std():.4f}")

print("\nPer-run GW 1-NN times (seconds):")
print(all_gw_times)

print(f"\nAverage GW 1-NN runtime per run: {all_gw_times.mean():.2f} s")
print(f"Std of GW 1-NN runtime: {all_gw_times.std():.2f} s")

print(f"\nTotal GW 1-NN runtime over all {n_repeats_gw} runs: {all_gw_times.sum():.2f} s")




###################################################################################################
## GW DICTIONARY LEARNING BASELINE + k-NN ON UNMIXING WEIGHTS ####################################
###################################################################################################
print("\n=== GW dictionary learning baseline (on selected MNIST point clouds) ===")

# -------------------------------------------------------------------
# 1) Learn GW dictionary on these Cs_list
# -------------------------------------------------------------------
D_dict = 3         # number of atoms in the dictionary
nt = 20            # number of nodes per atom
q = ot.unif(nt)    # uniform reference measure on atoms
reg = 0.0          # optional sparsity regularization on unmixings

print("\nLearning GW dictionary...")

t0_dict_learn = time.perf_counter()

Cdict_GW, log_dict = gromov_wasserstein_dictionary_learning(
    Cs=dist_matrices,
    D=D_dict,
    nt=nt,
    ps=measures,
    q=q,
    epochs=10,
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
# 2) Compute unmixing weights w_s for each sample (GW linear unmixing)
# -------------------------------------------------------------------
print("\nComputing unmixing weights (embeddings) for each sample...")

unmixings = []
recon_errors = []

t0_unmix = time.perf_counter()

for C_s, p_s in zip(dist_matrices, measures):
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
# 3) Use unmixing weights as features for classification with k-NN
#    Cross-validated k selection + repeated train/test splits
# -------------------------------------------------------------------
print("\nCross-validated k-NN on GW-dictionary unmixing features")

dict_features = unmixings
dict_labels = label_selected.astype(int)

candidate_ks = [1, 3, 5, 7, 9]
n_repeats_dict = 20
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
gw1nn_total_time = all_gw_times.mean()

print(f"GW 1-NN (templates): mean acc = {gw1nn_mean_acc:.4f}, std = {gw1nn_std_acc:.4f}, "
      f"time (average among runs) = {gw1nn_total_time:.4f} s")

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
