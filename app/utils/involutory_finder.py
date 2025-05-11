import numpy as np
from itertools import product


def is_involutory(K, m):
    K_squared = np.matmul(K, K) % m
    identity = np.eye(K.shape[0], dtype=int) % m
    return np.array_equal(K_squared, identity)


def generate_even_involutory_matrix(n, m):
    """
    Generate a structured involutory matrix for even n using block matrix construction.
    K = [[A11, A12],
         [A21, A22]]
    where A11 = -A22 mod m
    """
    half_n = n // 2
    for _ in range(100):
        A22 = np.random.randint(0, m, size=(half_n, half_n))
        A12 = np.random.randint(0, m, size=(half_n, half_n))
        A21 = np.random.randint(0, m, size=(half_n, half_n))
        A11 = (-A22) % m
        K = np.block([[A11, A12], [A21, A22]]) % m
        if is_involutory(K, m):
            return K
    return None


def generate_odd_involutory_matrix(n, m):
    """
    Generate a diagonal involutory matrix for odd n.
    Diagonal elements x must satisfy x^2 ≡ 1 mod m.
    """
    values = [x for x in range(m) if (x * x) % m == 1]
    if not values:
        return None
    diag = np.random.choice(values, size=n)
    return np.diag(diag) % m


def generate_involutory_matrix(n, m, max_attempts=100):
    if n % 2 == 0:
        return generate_even_involutory_matrix(n, m)
    else:
        return generate_odd_involutory_matrix(n, m)


def generate_all_diagonal_involutory(n, m):
    """
    Generate all diagonal involutory matrices where each diagonal element x satisfies x^2 ≡ 1 mod m.
    """
    values = [x for x in range(m) if (x * x) % m == 1]
    matrices = []
    for diag in product(values, repeat=n):
        K = np.diag(diag)
        matrices.append(K)
    return matrices


def generate_all_involutory_matrices(n, mod=26, max_matrices=10):
    """
    Generate a list of involutory matrices using structured methods for better performance.
    """
    found = []
    attempts = 0
    max_attempts = max_matrices * 200

    while len(found) < max_matrices and attempts < max_attempts:
        if n % 2 == 0:
            K = generate_even_involutory_matrix(n, mod)
        else:
            K = generate_odd_involutory_matrix(n, mod)
        if K is not None and not any(np.array_equal(K, f) for f in found):
            found.append(K)
        attempts += 1

    return found


def construct_from_user_blocks(n, m, A22, A12, max_attempts=1000):
    """
    Construct an involutory matrix using user-provided A22 and A12 blocks.
    Tries random A21 blocks to complete the matrix.
    """
    half_n = n // 2
    A11 = (-A22) % m
    for _ in range(max_attempts):
        A21 = np.random.randint(0, m, size=(half_n, half_n))
        K = np.block([[A11, A12], [A21, A22]]) % m
        if is_involutory(K, m):
            return K
    return None
