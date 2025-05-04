import streamlit as st
import numpy as np
from cipher_logic import encrypt, decrypt, mod_matrix_inverse
from utils.hillpp_cipher import hillpp_encrypt, hillpp_decrypt
from utils.involutory_finder import generate_involutory_matrix, is_involutory

st.set_page_config(page_title="🔐 Hill Cipher++", layout="wide")
st.title("🔐 Hill++ Cipher Interface")

modulus = st.number_input("Modulus (m):", min_value=2, max_value=100, value=26, step=1)
gamma = st.number_input("Gamma (secret multiplier γ):", min_value=1, max_value=100, value=3)

block_size = st.number_input("Matrix size (n x n):", min_value=2, max_value=6, value=2, step=1)

# Seed β Input
st.markdown("#### Seed β (initial vector of length n):")
beta = []
beta_cols = st.columns(block_size)
for i in range(block_size):
    beta.append(beta_cols[i].number_input(f"β[{i}]", min_value=0, max_value=modulus-1, value=1, key=f"beta_{i}"))

# Matrix Input
st.markdown("#### Involutory Matrix (n x n):")
key_matrix = np.zeros((block_size, block_size), dtype=int)
for i in range(block_size):
    row = st.columns(block_size)
    for j in range(block_size):
        key_matrix[i][j] = row[j].number_input(
            f"K[{i},{j}]", min_value=0, max_value=modulus-1,
            value=3 if i == j else 0, key=f"key_{i}_{j}")

st.write("Current Key Matrix:")
st.write(key_matrix)

if is_involutory(key_matrix, modulus):
    st.success("✅ The key matrix is involutory (K² ≡ I mod m).")
else:
    st.warning("⚠️ The matrix is not involutory. Hill++ decryption may fail.")

# Input text (shared)
st.markdown("#### Input Text (plaintext or ciphertext):")
text = st.text_input("Enter your message (A–Z only):", "HELLO")

# Side-by-side Encrypt / Decrypt
left, right = st.columns(2)

with left:
    st.markdown("### 🔐 Encrypt")
    if st.button("▶️ Run Encryption"):
        try:
            C_blocks, encrypted_text = hillpp_encrypt(text, key_matrix, gamma, beta, modulus)
            st.success(f"Encrypted text: {encrypted_text}")
            st.write("🔐 Cipher blocks:")
            st.write(C_blocks)
        except Exception as e:
            st.error(f"❌ Encryption error: {e}")

with right:
    st.markdown("### 🔓 Decrypt")
    if st.button("▶️ Run Decryption"):
        try:
            P_blocks, decrypted_text = hillpp_decrypt(text, key_matrix, gamma, beta, modulus)
            st.success(f"Decrypted text: {decrypted_text}")
            st.write("🔓 Plaintext blocks:")
            st.write(P_blocks)
        except Exception as e:
            st.error(f"❌ Decryption error: {e}")
