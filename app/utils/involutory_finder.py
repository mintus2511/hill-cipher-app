# app/utils/involutory_finder.py

import numpy as np
from itertools import product

def is_involutory(K, m):
    K_squared = np.matmul(K, K) % m
    identity = np.eye(K.shape[0], dtype=int) % m
    return np.array_equal(K_squared, identity)

def generate_even_involutory_matrix(n, m):
    half_n = n // 2
    A22 = np.random.randint(0, m, size=(half_n, half_n))
    A12 = np.random.randint(0, m, size=(half_n, half_n))
    A11 = (-A22) % m
    A21 = np.random.randint(0, m, size=(half_n, half_n))
    K = np.block([[A11, A12], [A21, A22]]) % m
    return K if is_involutory(K, m) else None

def generate_odd_involutory_matrix(n, m):
    K = np.zeros((n, n), dtype=int)
    for i in range(n):
        K[i, i] = 1 if np.random.randint(0, 2) == 0 else m - 1
    return K

def generate_involutory_matrix(n, m, max_attempts=100):
    if n % 2 == 0:
        for _ in range(max_attempts):
            K = generate_even_involutory_matrix(n, m)
            if K is not None:
                return K
    else:
        return generate_odd_involutory_matrix(n, m)

def generate_all_diagonal_involutory(n, m):
    values = [x for x in range(m) if (x * x) % m == 1]
    matrices = []
    for diag in product(values, repeat=n):
        K = np.diag(diag)
        matrices.append(K)
    return matrices

def generate_all_involutory_matrices(n, m, max_matrices=100):
    matrices = generate_all_diagonal_involutory(n, m)
    if n % 2 == 0:
        count = 0
        for _ in range(max_matrices):
            K = generate_even_involutory_matrix(n, m)
            if K is not None and not any(np.array_equal(K, mat) for mat in matrices):
                matrices.append(K)
                count += 1
    return matrices

def construct_from_user_blocks(n, m, A22, A12, max_attempts=1000):
    half_n = n // 2
    A11 = (-A22) % m
    for _ in range(max_attempts):
        A21 = np.random.randint(0, m, size=(half_n, half_n))
        K = np.block([[A11, A12], [A21, A22]]) % m
        if is_involutory(K, m):
            return K
    return None
