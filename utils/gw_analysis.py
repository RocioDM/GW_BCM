import numpy as np
#from scipy.optimize import minimize
import ot
import cvxpy as cp


#### GW Analysis based on a Gradient Method via fixed-point #######################################


def get_lambdas(matrix_temp_list, measure_temp_list, matrix_input, measure_input):
    '''
    Computes the barycentric weights (lambda_1,...,lambda_S), where S is the number of templates.
    Computes a candidate for a barycenter matrix associated with the barycenter weights using one iteration
    of the forward GW-barycenter problem.

    Input:
      matrix_temp_list: List of S template matrices (Ns x Ns) representing different dissimilarity matrices.
      measure_temp_list: List of S probability vectors (length Ns), representing probability measures of the S templates.
      matrix_input: (M x M) matrix representing the dissimilarity matrix to analyze (array).
      measure_input: Probability vector of length M (array)

    Output:
      lambdas: Vector of weights, one for each template (S elements). No constraints are imposed. (array)
      matrix_output = synthesized matrix with GW barycentric coordinates given by the lambdas vector. (array)
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

    return matrix_output, lambdas



### adding constraints
def get_lambdas_constraints(matrix_temp_list, measure_temp_list, matrix_input, measure_input):
  """
  Computes the barycentric weights (lambda_1,...,lambda_S) in the (S-1)-dimensional simples,
  where S is the number of templates.
  Computes a candidate for a barycenter matrix associated with the barycenter weights using one iteration
  of the forward GW-barycenter problem.


  Input:
      matrix_temp_list: List of S template matrices (Ns x Ns) representing different dissimilarity matrices.
      measure_temp_list: List of S probability vectors (length Ns), representing probability measures of the S templates.
      matrix_input: (M x M) matrix representing the dissimilarity matrix to analyze.
      measure_input: Probability vector of length M.

  Output:
      lambdas: Vector of weights, one for each template (S elements). Constraints are imposed, that is,
      lambdas belongs to the (S-1) probability simplex. (array)
      matrix_output = synthesized matrix with GW barycentric coordinates given by the lambdas vector. (array)
  """

  S = len(matrix_temp_list)

  pi_list = []
  F_list = []

  D_recon = np.zeros_like(matrix_input)

  Q = (measure_input.reshape(-1,1) @ measure_input.reshape(1,-1))
  Q = 1. / Q

  for s in range(S):
      pi_s = ot.gromov.gromov_wasserstein(matrix_temp_list[s],matrix_input,measure_temp_list[s],measure_input)
      pi_list.append(pi_s)
      F_s = Q * (pi_s.T @ matrix_temp_list[s] @ pi_s)
      F_list.append(F_s)

  K = np.zeros((S,S))
  b = np.zeros(S)
  for i in range(S):
      b[i] = np.trace(matrix_input @ F_list[i])
      for j in range(S):
          K[i,j] = np.trace(F_list[i].T @ F_list[j])


  A = np.ones([1,S])
  G = - np.eye(S)
  h = np.zeros(S)

  x = cp.Variable(S)
  constraints = [
    A @ x == 1,  # Equality constraint
    G @ x <= h   # Inequality constraint
  ]
  objective = cp.Minimize((1/2) * cp.quad_form(x, A) -b @ x)

  # Solve the quadratic optimization problem
  prob = cp.Problem(objective, constraints)

  prob.solve()

  lambdas = x.value

  # Compute the synthesized output matrix
  matrix_output = np.zeros_like(matrix_input)
  for s in range(S):
    D_recon += lambdas[s] * F_list[s]
    matrix_output = D_recon


  return matrix_output, lambdas




#### GW Analysis based on a Gradient Method via blow-ups ##########################################

def get_lambdas_blowup(X, Y, q):
    '''
    Input:
     - param X: list of template matrices after blowup
     - param Y: input matrix after blow up (array)
     - param q: input measure (probability vector) after blow up (array)
    Output:
     -lambdas_recon = vector of weights, as many as number of templates, representing GW barycentric coordinates. (array)
     -Y_recon = synthesized matrix ac convex combination of the matrices in the list of templates X
      with GW barycentric coordinates given by the lambdas_recon vector. (array)
    '''
    S = len(X) # number of templates

    ## Cost function to be used in the GW - analysis problem via blow-ups
    A = np.zeros((S, S))

    for s in range(S):
        for r in range(S):
            #A[s, r] = np.sum(q * (X[s] - Y) @ (X[r] - Y).T @ q)
            A[s, r] = np.trace((X[s] - Y) @ np.diag(q) @ (X[r] - Y).T @ np.diag(q))

    lambda_var = cp.Variable(S)
    objective = 0.5 * cp.quad_form(lambda_var, A)
    constraints = [lambda_var >= 0, cp.sum(lambda_var) == 1]
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.OSQP)  # Or ECOS

    lambdas_recon = lambda_var.value



    # # Objective function:
    # def objective(lambda_vec):
    #     return 0.5 * lambda_vec.T @ A @ lambda_vec
    #
    # # Constraints: Sum of λ = 1 and λ ≥ 0
    # constraints = [
    #     {"type": "eq", "fun": lambda lambda_vec: np.sum(lambda_vec) - 1},
    # ]
    #
    # bounds = [(0, 1) for _ in range(S)]
    #
    # # Initial guess (uniform distribution in the simplex)
    # initial_guess = np.ones(S) / S
    #
    # # Optimization
    # result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    #
    # if result.success:
    #     lambdas_recon = result.x
    #     # Reconstruct a barycenter
    #     Y_recon = np.zeros_like(X[0])
    #     for i in range(S):
    #         Y_recon += lambdas_recon[i] * X[i]
    #     return Y_recon, lambdas_recon
    # else:
    #     raise ValueError("Optimization failed.")

    Y_recon = np.zeros_like(X[0])
    for i in range(S):
        Y_recon += lambdas_recon[i] * X[i]
    return Y_recon, lambdas_recon




def blow_up(matrix_temp_list, measure_temp_list, B, b):
    """
      blow_up computes the blow-up and realignment ensuring that the new networks
      have all the same size
      input:
        matrix_temp_list: list of S arrays of different dimensions (Ns,Ns),
                each one represents a (Ns x Ns) matrix that is one of the S different dissimilarity matrices of the S templates.
        measure_temp_list: list of S arrays of different dimensions Ns,
                each one represents a probability vector in R^Ns that is one of the S probability measures of the S templates.
        B: matrix representing a dissimilarity matrix you want to analyze (array)
        b: probability vector (array)
      output:
        temp_blow_up = new blow-up list of S arrays of the same dimensions
        B = new blow-up matrix (array)
        b = blow up probability vector (array)
      """
    S = len(matrix_temp_list) # number of templates
    temp_blow_up = []
    for i in range(S):
        X = matrix_temp_list[i]
        p = measure_temp_list[i]

        pi = ot.gromov.gromov_wasserstein(X, B, p, b)

        row_indices, col_indices = np.nonzero(pi)

        # Combine row and column indices into coordinate pairs
        non_zero_coords = np.array(list(zip(row_indices, col_indices)))
        v_x = non_zero_coords[:, 0]
        v_y = non_zero_coords[:, 1]
        # print('size of blow up: ', len(v_x))

        b = pi[v_x, v_y]

        V_1, V_2 = np.meshgrid(v_y, v_y)
        B_tilde = B[V_1, V_2]
        B = B_tilde.copy()

        A_1, A_2 = np.meshgrid(v_x, v_x)
        X_tilde = X[A_1, A_2]
        X = X_tilde.copy()

        temp_blow_up.append(X)

        for j in range(i):
            Z = temp_blow_up[j].copy()
            temp_blow_up[j] = Z[V_1, V_2]

    return B, b, temp_blow_up