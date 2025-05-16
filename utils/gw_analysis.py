import numpy as np  # linear algebra
from scipy.optimize import minimize
import ot
import cvxpy as cp

def get_lambdas(matrix_temp_list, measure_temp_list, matrix_input, measure_input):
    '''
    Computes the barycentric weights (lambda_1,...,lambda_S), where S is the number of templates.
    Computes a candidate for a barycenter matrix associated with the barycenter weights using one iteration of the forward GW-barycenter problem.
      See for example remark 2.9 or equation 11.

    Input:
      matrix_temp_list: List of S template matrices (Ns x Ns) representing different dissimilarity matrices.
      measure_temp_list: List of S probability vectors (length Ns), representing probability measures of the S templates.
      matrix_input: (M x M) matrix representing the dissimilarity matrix to analyze.
      measure_input: Probability vector of length M.

    Output:
      lambdas: Vector of weights, one for each template (S elements). These are not necessarily non-negative.
      matrix_output: Synthesized dissimilarity matrix.
    '''

    S = len(matrix_temp_list)  # Number of template matrices

    pi_list = []  # List to store Gromov-Wasserstein transport plans
    F_list = []  # List to store transformed matrices

    # Compute Q matrix (inverse of the outer product of measure_input)
    Q = (measure_input.reshape(-1, 1) @ measure_input.reshape(1, -1))
    Q = 1. / Q  # Element-wise inverse

    # Compute Gromov-Wasserstein transport maps and one iteration of the forward GW-barycenter problem
    for s in range(S):
        # Compute optimal transport plan (pi_s) using Gromov-Wasserstein transport
        pi_s = ot.gromov.gromov_wasserstein(matrix_temp_list[s], matrix_input,
                                            measure_temp_list[s], measure_input)
        pi_list.append(pi_s)

        # Compute F_s transformation using Q and pi_s
        F_s = Q * (pi_s.T @ matrix_temp_list[s] @ pi_s)  # Element-wise multiplication
        F_list.append(F_s)

    # Set up and solve the linear problem (K@lambdas = b) for the vector of weights lambdas
    # Construct K matrix and b vector for least squares problem
    K = np.zeros((S, S))
    b = np.zeros(S)

    for i in range(S):
        b[i] = np.trace(matrix_input @ F_list[i])  # Compute b_i
        for j in range(S):
            K[i, j] = np.trace(F_list[i] @ F_list[j])  # Compute K_ij

    # Augment K with an additional row and column for sum-to-one constraint
    K_aug = np.hstack([K, -0.5 * np.ones(S).reshape(-1, 1)])
    last_row = np.ones(K_aug.shape[1]).reshape(1, -1)
    last_row[0, -1] = 0
    K_aug = np.vstack([K_aug, last_row])

    # Augment b with the constraint that lambdas sum to 1
    b_aug = np.hstack([b, [1]])

    # Solve for lambdas using the linear system K_aug * lambdas = b_aug
    lambdas = np.linalg.solve(K_aug, b_aug)

    # Extract lambda values (excluding the last auxiliary value corresponding to the lagrange multiplier of the sum-to-one constraint)
    lambdas = lambdas[:-1]

    # Compute the synthesized output matrix
    matrix_output = np.zeros_like(matrix_input)

    for s in range(S):
        matrix_output += lambdas[s] * F_list[s]  # Weighted sum of transformed matrices
        matrix_input = matrix_output  # Update matrix_input (although this might be redundant)

    return matrix_output, lambdas


####

def get_lambdas_no_constraint(matrix_temp_list, measure_temp_list, matrix_input, measure_input):
    '''
    Computes the barycentric weights (lambda_1,...,lambda_S), where S is the number of templates.
    Computes a candidate for a barycenter matrix associated with the barycenter weights using one iteration of the forward GW-barycenter problem.
      See for example remark 2.9 or equation 11.

    Input:
      matrix_temp_list: List of S template matrices (Ns x Ns) representing different dissimilarity matrices.
      measure_temp_list: List of S probability vectors (length Ns), representing probability measures of the S templates.
      matrix_input: (M x M) matrix representing the dissimilarity matrix to analyze.
      measure_input: Probability vector of length M.

    Output:
      lambdas: Vector of weights, one for each template (S elements). No constraints are imposed.
    '''

    S = len(matrix_temp_list)  # Number of template matrices

    pi_list = []  # List to store Gromov-Wasserstein transport plans
    F_list = []  # List to store transformed matrices

    # Compute Q matrix (inverse of the outer product of measure_input)
    Q = (measure_input.reshape(-1, 1) @ measure_input.reshape(1, -1))
    Q = 1. / Q  # Element-wise inverse

    # Compute Gromov-Wasserstein transport maps and one iteration of the forward GW-barycenter problem
    for s in range(S):
        # Compute optimal transport plan (pi_s) using Gromov-Wasserstein transport
        pi_s = ot.gromov.gromov_wasserstein(matrix_temp_list[s], matrix_input,
                                            measure_temp_list[s], measure_input)
        pi_list.append(pi_s)

        # Compute F_s transformation using Q and pi_s
        F_s = Q * (pi_s.T @ matrix_temp_list[s] @ pi_s)  # Element-wise multiplication
        F_list.append(F_s)

    # Set up and solve the linear problem (K@lambdas = b) for the vector of weights lambdas
    # Construct K matrix and b vector for least squares problem
    K = np.zeros((S, S))
    b = np.zeros(S)

    for i in range(S):
        b[i] = np.trace(matrix_input @ F_list[i])  # Compute b_i
        for j in range(S):
            K[i, j] = np.trace(F_list[i] @ F_list[j])  # Compute K_ij

    # Solve for lambdas in the linear system K * lambdas = b
    lambdas = np.linalg.solve(K, b)

    # Compute the synthesized output matrix
    matrix_output = np.zeros_like(matrix_input)

    for s in range(S):
        matrix_output += lambdas[s] * F_list[s]  # Weighted sum of transformed matrices
        matrix_input = matrix_output  # Update matrix_input (although this might be redundant)

    return matrix_output, lambdas



####

def get_lambdas2(C_list, p_list, D, q):
  '''
  overdetermined linear system
  '''
  S = len(C_list)

  pi_list = []
  F_list = []

  Q = (q.reshape(-1,1) @ q.reshape(1,-1))
  Q = 1. / Q

  for s in range(S):
      pi_s = ot.gromov.gromov_wasserstein(C_list[s],D,p_list[s],q)
      pi_list.append(pi_s)
      F_s = Q * (pi_s.T @ C_list[s] @ pi_s)
      F_list.append(F_s)

  K = np.zeros((S,S))
  b = np.zeros(S)
  for i in range(S):
      b[i] = np.trace(D @ F_list[i])
      for j in range(S):
          K[i,j] = np.trace(F_list[i] @ F_list[j])

  last_row = np.ones(K.shape[1]).reshape(1,-1)
  K_aug = np.vstack([K,last_row])
  b_aug = np.hstack([b, [1]])

  lambdas, residuals, rank, s= np.linalg.lstsq(K_aug, b_aug, rcond=None)

  D_recon = np.zeros_like(D)

  for s in range(S):
    D_recon += lambdas[s] * F_list[s]
    D = D_recon

  return D, lambdas

## We still need to add the restriction lambdas[s] >=0


def get_lambdas3(C_list, p_list, D, q):
  """
  get_lambdas computes the barycentric weights (lambda_1,...,lambda_S) from the simplified GW-Barycenter Analysis Problem
  and reconstruct a matrix that is "close" to a barycentric matrix
  input:
    S: number of templates
    C_list: list of S arrays of different dimensions (Ns,Ns),
            each one represents a (Ns x Ns) matrix that is one of the S different dissimilarity matrices of the S templates
    p_list: list of S arrays of different dimensions Ns,
            each one represents a probability vector in R^Ns that is one of the S probability measures of the S templates
    D: matrix representing a dissimilarity matrix you want to analyze (array)
    q: probability vector of size of D (array)
  output:
  lambdas = vector of weights, as many as number of templates
  D = new synthesize matrix
  """

  S = len(C_list)

  pi_list = []
  F_list = []

  D_recon = np.zeros_like(D)

  Q = (q.reshape(-1,1) @ q.reshape(1,-1))
  Q = 1. / Q

  for s in range(S):
      pi_s = ot.gromov.gromov_wasserstein(C_list[s],D,p_list[s],q)
      pi_list.append(pi_s)
      F_s = Q * (pi_s.T @ C_list[s] @ pi_s)
      F_list.append(F_s)

  K = np.zeros((S,S))
  b = np.zeros(S)
  for i in range(S):
      b[i] = np.trace(D @ F_list[i])
      for j in range(S):
          K[i,j] = np.trace(F_list[i] @ F_list[j])

  P = K.T @ K
  v = - b.T @ K
  A = np.ones([1,S])
  G = - np.eye(S)
  h = np.zeros(S)

  x = cp.Variable(S)
  constraints = [
    A @ x == 1,  # Equality constraint
    G @ x <= h   # Inequality constraint
  ]
  objective = cp.Minimize((1/2) * cp.quad_form(x, P) + v.T @ x)

  # Solve the problem
  prob = cp.Problem(objective, constraints)
  prob.solve()

  lambdas = x.value

  for s in range(S):
    D_recon += lambdas[s] * F_list[s]
    D = D_recon


  return D, lambdas



#### GW Analysis based on a Gradient Method via blow-ups


## Cost function to be used in the GW - analysis problem via blow-ups

def get_lambdas_blowup(X, Y, q):
    '''
    Input:
     - param X: list of template matrices after blowup
     - param Y: input matrix after blow up
     - param q: input measure (probability vector) after blow up
    Output: vector of weights, as many as number of templates
    '''
    S = len(X) # number of templates

    ## Cost function to be used in the GW - analysis problem via blow-ups
    A = np.zeros((S, S))

    for s in range(S):
        for r in range(S):
            A[s, r] = np.sum(q * (X[s] - Y) @ (X[r] - Y).T @ q)
            #A[s, r] = np.trace((X[s] - Y) @ np.diag(q) @ (X[r] - Y).T @ np.diag(q))

    # Objective function: 0.5 * λ^T A λ
    def objective(lambda_vec):
        return 0.5 * lambda_vec.T @ A @ lambda_vec

    # Constraints: Sum of λ = 1 and λ ≥ 0
    constraints = [
        {"type": "eq", "fun": lambda lambda_vec: np.sum(lambda_vec) - 1},
    ]

    bounds = [(0, 1) for _ in range(S)]

    # Initial guess (uniform distribution in the simplex)
    initial_guess = np.ones(S) / S

    # Optimization
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

    if result.success:
        lambdas_recon = result.x
        # Reconstruct a barycenter
        Y_recon = np.zeros_like(X[0])
        for i in range(S):
            Y_recon += lambdas_recon[i] * X[i]
        return Y_recon, lambdas_recon
    else:
        raise ValueError("Optimization failed.")



# def cost_analysis_blowup(X, Y):
#     '''
#     Input:
#      - param X: list of template matrices
#      - param Y: input matrix
#     :return: cost matrix (num templates)x(number of templates)
#     '''
#     S = len(X) # number of templates
#     A = np.zeros((S, S))
#
#     for s in range(S):
#         for r in range(S):
#             A[s, r] = np.trace((X[s] - Y) @ (X[r] - Y).T)
#
#     return A
#
# def get_lambdas_blowup(A):
#     S = A.shape[0]
#
#     # Objective function: 0.5 * λ^T A λ
#     def objective(lambda_vec):
#         return 0.5 * lambda_vec.T @ A @ lambda_vec
#
#     # Constraints: Sum of λ = 1 and λ ≥ 0
#     constraints = [
#         {"type": "eq", "fun": lambda lambda_vec: np.sum(lambda_vec) - 1},
#     ]
#
#     bounds = [(0, 1) for _ in range(S)]
#
#     # Initial guess (uniform distribution in the simplex)
#     initial_guess = np.ones(S) / S
#
#     # Optimization
#     result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
#
#     if result.success:
#         return result.x
#     else:
#         raise ValueError("Optimization failed.")