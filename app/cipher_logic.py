import numpy as np
from sympy import Matrix

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def preprocess_text(text, block_size):
    text = ''.join(filter(str.isalpha, text.upper()))
    if len(text) % block_size != 0:
        text += 'X' * (block_size - len(text) % block_size)
    return text

def text_to_numbers(text):
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [ALPHABET.index(char) for char in text if char in ALPHABET]

def numbers_to_text(numbers):
    return ''.join(ALPHABET[num % 26] for num in numbers)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def is_invertible_matrix(matrix, mod):  # Renamed for compatibility
    det = int(round(np.linalg.det(matrix)))
    return gcd(det, mod) == 1

def mod_matrix_inverse(matrix, mod):
    try:
        sympy_matrix = Matrix(matrix)
        return np.array(sympy_matrix.inv_mod(mod)).astype(int)
    except:
        raise ValueError(f"Matrix is not invertible under mod {mod}")

def encrypt(plaintext, key_matrix, mod=26):
    block_size = key_matrix.shape[0]
    
    # Làm sạch: chỉ giữ ký tự A–Z
    clean_text = ''.join(filter(str.isalpha, plaintext.upper()))
    if not clean_text:
        raise ValueError("Input must contain at least one valid A–Z character.")

    # Thêm padding nếu cần
    padded_text = preprocess_text(clean_text, block_size)
    
    # Convert to numbers
    numbers = text_to_numbers(padded_text)
    ciphertext = []

    for i in range(0, len(numbers), block_size):
        block = np.array(numbers[i:i + block_size])
        enc_block = key_matrix.dot(block) % mod
        ciphertext.extend(enc_block)

    return numbers_to_text(ciphertext)


def decrypt(ciphertext, key_matrix, mod=26):
    block_size = key_matrix.shape[0]
    clean_text = ''.join(filter(str.isalpha, ciphertext.upper()))
    if not clean_text:
        raise ValueError("Ciphertext must contain A–Z characters only.")

    numbers = text_to_numbers(clean_text)
    inverse_key = mod_matrix_inverse(key_matrix, mod)
    plaintext = []

    for i in range(0, len(numbers), block_size):
        block = np.array(numbers[i:i + block_size])
        dec_block = inverse_key.dot(block) % mod
        plaintext.extend(dec_block)

    return numbers_to_text(plaintext)
