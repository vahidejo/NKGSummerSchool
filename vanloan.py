import numpy as np
from scipy.linalg import expm

def numeval(F, G, dt):

    # Matrix size
    [m, n] = F.shape

    # Build matrix (Van Loan)
    A1 = np.append(-F, G@G.T, axis=1)
    A2 = np.append(np.zeros([m, n]), F.T, axis=1)
    A = np.append(A1, A2, axis=0)*dt

    # Compute matrix exponential
    B = expm(A)

    # Extract phi and Q
    phi = B[m:m*2, n:n*2].T

    Q = phi@B[0:m, n:n*2]

    return phi, Q
