import numpy as np

def qr_factorization(A):
    """Performs QR factorization of matrix A using the Gram-Schmidt process."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j]
        
        # Subtract the projections of v onto the previous q's
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        # Normalize the vector to create the orthonormal basis
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R

# Example usage
A = np.array([[1, 2, 1, 0], [1, 1, 2, 1], [1, 0, 1, 2], [0, 1, 1, 1]], dtype=float)
Q, R = qr_factorization(A)

print("Q Matrix:")
print(Q)
print("\nR Matrix:")
print(R)