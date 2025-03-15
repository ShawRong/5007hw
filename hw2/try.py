import numpy as np

def rank_approximation(matrix, k):
    """
    Calculate the rank-k approximation of the given matrix using SVD.

    Parameters:
    matrix (numpy.ndarray): The input matrix to approximate.
    k (int): The rank for the approximation.

    Returns:
    numpy.ndarray: The rank-k approximation of the matrix.
    """
    # Perform SVD
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)

    # Keep only the top k singular values and vectors
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]

    # Reconstruct the rank-k approximation
    rank_k_approx = np.dot(U_k, np.dot(S_k, VT_k))
    return rank_k_approx

# Example usage
if __name__ == "__main__":
    # Create a sample matrix
    A = np.array([[2, 1, 1],
                  [1, 2, 1],
                  [1, 1, 2]])

    # Set the desired rank
    k = 1

    # Calculate the rank-k approximation
    approx = rank_approximation(A, k)
    print("Original Matrix:\n", A)
    print("Rank-k Approximation:\n", approx)