import numpy as np

def generate_involutory_matrix(size, mod=26):
    """
    Generates an involutory matrix A such that A * A ≡ I mod 'mod'
    """
    attempts = 0
    max_attempts = 1000  # Prevent infinite loop

    while attempts < max_attempts:
        A = np.random.randint(0, mod, size=(size, size))
        A_squared = np.mod(np.dot(A, A), mod)
        if np.array_equal(A_squared, np.identity(size, dtype=int) % mod):
            return A
        attempts += 1

    raise ValueError(f"Failed to find involutory matrix after {max_attempts} attempts.")
