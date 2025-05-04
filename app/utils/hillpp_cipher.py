import numpy as np

def text_to_vector(text, block_size):
    text = text.upper().replace(" ", "")
    nums = [ord(c) - ord('A') for c in text if c.isalpha()]
    while len(nums) % block_size != 0:
        nums.append(0)  # pad with A (0)
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
