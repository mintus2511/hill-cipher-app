import numpy as np

def text_to_vector(text, block_size):
    text = ''.join(filter(str.isalpha, text.upper()))
    nums = [ord(c) - ord('A') for c in text]
    if len(nums) % block_size != 0:
        nums += [23] * (block_size - len(nums) % block_size)  # pad with 'X'
    return np.array(nums).reshape(-1, block_size)

def vector_to_text(vec):
    return ''.join(chr(int(round(num)) % 26 + ord('A')) for num in vec.flatten())

def hillpp_encrypt_verbose(plaintext, K, gamma, beta, m=26):
    steps = []
    block_size = K.shape[0]
    P = text_to_vector(plaintext, block_size)
    C = []
    prev_C = np.array(beta) % m
    steps.append(f"Input text: {plaintext}")
    steps.append(f"Preprocessed numeric matrix P:\n{P}")
    steps.append(f"Initial beta (\u03b2): {prev_C}")

    for i in range(P.shape[0]):
        Pi = P[i, :]
        RMK = (gamma * prev_C) % m
        Ci = (np.dot(K, Pi) + RMK) % m
        steps.append(f"--- Block {i+1} ---")
        steps.append(f"P[{i}] = {Pi}")
        steps.append(f"R * beta = {gamma} * {prev_C} % {m} = {RMK}")
        steps.append(f"C[{i}] = (K @ P[{i}] + RMK) % {m} = {Ci}")
        C.append(Ci)
        prev_C = Ci

    result = vector_to_text(np.array(C))
    steps.append(f"Final encrypted text: {result}")
    return steps, result

def hillpp_decrypt_verbose(ciphertext, K, gamma, beta, m=26):
    steps = []
    block_size = K.shape[0]
    C = text_to_vector(ciphertext, block_size)
    P = []
    prev_C = np.array(beta) % m
    inv_K = np.linalg.inv(K).astype(int) % m
    steps.append(f"Input cipher text: {ciphertext}")
    steps.append(f"Cipher numeric matrix C:\n{C}")
    steps.append(f"Initial beta (\u03b2): {prev_C}")
    steps.append(f"Inverse of K mod {m}:\n{inv_K}")

    for i in range(C.shape[0]):
        Ci = C[i, :]
        RMK = (gamma * prev_C) % m
        Pi = np.dot(inv_K, (Ci - RMK)) % m
        steps.append(f"--- Block {i+1} ---")
        steps.append(f"C[{i}] = {Ci}")
        steps.append(f"R * beta = {gamma} * {prev_C} % {m} = {RMK}")
        steps.append(f"P[{i}] = invK @ (C[{i}] - RMK) % {m} = {Pi}")
        P.append(Pi)
        prev_C = Ci

    result = vector_to_text(np.array(P))
    steps.append(f"Final decrypted text: {result}")
    return steps, result
