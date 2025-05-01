
import numpy as np
from sympy import Matrix

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def preprocess_text(text, block_size):
    text = ''.join(filter(str.isalpha, text.upper()))
    if len(text) % block_size != 0:
        text += 'X' * (block_size - len(text) % block_size)
    return text

def text_to_numbers(text):
    return [ALPHABET.index(char) for char in text]

def numbers_to_text(numbers):
    return ''.join(ALPHABET[num % 26] for num in numbers)

def is_invertible(matrix, mod):
    det = int(round(np.linalg.det(matrix)))
    return gcd(det, mod) == 1

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def mod_matrix_inverse(matrix, mod):
    try:
        sympy_matrix = Matrix(matrix)
        return np.array(sympy_matrix.inv_mod(mod)).astype(int)
    except:
        raise ValueError("Matrix is not invertible under mod {}".format(mod))

def encrypt(plaintext, key_matrix, mod=26):
    block_size = key_matrix.shape[0]
    plaintext = preprocess_text(plaintext, block_size)
    numbers = text_to_numbers(plaintext)
    ciphertext = []

    for i in range(0, len(numbers), block_size):
        block = np.array(numbers[i:i + block_size])
        enc_block = key_matrix.dot(block) % mod
        ciphertext.extend(enc_block)

    return numbers_to_text(ciphertext)

def decrypt(ciphertext, key_matrix, mod=26):
    block_size = key_matrix.shape[0]
    numbers = text_to_numbers(ciphertext)
    inverse_key = mod_matrix_inverse(key_matrix, mod)
    plaintext = []

    for i in range(0, len(numbers), block_size):
        block = np.array(numbers[i:i + block_size])
        dec_block = inverse_key.dot(block) % mod
        plaintext.extend(dec_block)

    return numbers_to_text(plaintext)
