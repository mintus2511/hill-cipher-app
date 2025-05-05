import numpy as np

def text_to_vector(text, block_size):
    return np.array(nums).reshape(-1, block_size)

def vector_to_text(vec):
    return ''.join(chr(int(round(num)) % 26 + ord('A')) for num in vec.flatten())

def hillpp_encrypt(plaintext, K, gamma, beta, m=26):
    block_size = K.shape[0]
    P = text_to_vector(plaintext, block_size)
    C = []
    prev_C = np.array(beta) % m

    for i in range(P.shape[0]):
        RMK = (gamma * prev_C) % m
        Pi = P[i, :]
        Ci = (np.dot(K, Pi) + RMK) % m
        C.append(Ci)
        prev_C = Ci

    return np.array(C), vector_to_text(np.array(C))

def hillpp_decrypt(ciphertext, K, gamma, beta, m=26):
    block_size = K.shape[0]
    C = text_to_vector(ciphertext, block_size)
    P = []
    prev_C = np.array(beta) % m

    for i in range(C.shape[0]):
        RMK = (gamma * prev_C) % m
        Ci = C[i, :]
        Pi = np.dot(K, (Ci - RMK)) % m
        P.append(Pi)
        prev_C = Ci

    return np.array(P), vector_to_text(np.array(P))

def hillpp_encrypt_verbose(plaintext, K, gamma, beta, m=26):
    steps = []
    block_size = K.shape[0]
    plaintext = ''.join(filter(str.isalpha, plaintext.upper()))
    if len(plaintext) % block_size != 0:
        plaintext += 'X' * (block_size - len(plaintext) % block_size)

    nums = [ord(c) - ord('A') for c in plaintext]
    steps.append(f"1. Preprocessed text: {plaintext}")
    steps.append(f"2. Numeric values: {nums}")

    P = np.array(nums).reshape(-1, block_size)
    C = []
    prev_C = np.array(beta) % m

    for i in range(P.shape[0]):
        RMK = (gamma * prev_C) % m
        Pi = P[i, :]
        Ci = (np.dot(K, Pi) + RMK) % m
        steps.append(f"Block {i+1}:")
        steps.append(f"  Pi = {Pi}")
        steps.append(f"  RMK = γ * previous_C = {gamma} * {prev_C} ≡ {RMK} mod {m}")
        steps.append(f"  Ci = (K·Pi + RMK) mod {m} = {Ci}")
        C.append(Ci)
        prev_C = Ci

    result = ''.join(chr(int(round(n)) % 26 + ord('A')) for n in np.array(C).flatten())
    steps.append(f"Final ciphertext: {result}")
    return steps, result

def hillpp_decrypt_verbose(ciphertext, K, gamma, beta, m=26):
    steps = []
    block_size = K.shape[0]
    nums = [ord(c) - ord('A') for c in ciphertext]
    steps.append(f"1. Ciphertext input: {ciphertext}")
    steps.append(f"2. Numeric values: {nums}")

    C = np.array(nums).reshape(-1, block_size)
    P = []
    prev_C = np.array(beta) % m

    for i in range(C.shape[0]):
        RMK = (gamma * prev_C) % m
        Ci = C[i, :]
        Pi = np.dot(K, (Ci - RMK)) % m
        steps.append(f"Block {i+1}:")
        steps.append(f"  Ci = {Ci}")
        steps.append(f"  RMK = γ * previous_C = {gamma} * {prev_C} ≡ {RMK} mod {m}")
        steps.append(f"  Pi = K·(Ci - RMK) mod {m} = {Pi}")
        P.append(Pi)
        prev_C = Ci

    result = ''.join(chr(int(round(n)) % 26 + ord('A')) for n in np.array(P).flatten())
    steps.append(f"Final plaintext: {result}")
    return steps, result
