
import numpy as np
from itertools import product

def find_involutory_matrices(n=2, mod=26):
    identity = np.identity(n, dtype=int)
    found = []

    for flat in product(range(mod), repeat=n*n):
        A = np.array(flat).reshape((n, n))
        if np.array_equal((A @ A) % mod, identity):
            found.append(A)

    return found
