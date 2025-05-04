
import streamlit as st
import numpy as np
from cipher_logic import encrypt, decrypt, mod_matrix_inverse
from utils.involutory_finder import (
    generate_involutory_matrix,
    generate_all_involutory_matrices,
    construct_from_user_blocks,
    is_involutory
)
from utils.hillpp_cipher import hillpp_encrypt, hillpp_decrypt


def parse_text_matrix(text_rows, m):
    """Parses a list of text rows into a numpy matrix modulo m"""
    try:
        matrix = [[int(x) % m for x in row.strip().split()] for row in text_rows]
        return np.array(matrix, dtype=int)
    except Exception:
        return None

st.set_page_config(page_title="🔐 Hill Cipher++", layout="wide")
st.title("🔐 Hill Cipher++ Visualization")

block_size = st.selectbox("Matrix size (n x n)", [2, 3])

st.markdown("---")
st.subheader("🔑 Manual Key Matrix Input")

st.markdown(f"Enter your {block_size}×{block_size} key matrix below (mod 26):")

key_matrix = np.zeros((block_size, block_size), dtype=int)
for i in range(block_size):
    cols = st.columns(block_size)
    for j in range(block_size):
        key_matrix[i][j] = cols[j].number_input(
            f"Key[{i},{j}]", min_value=0, max_value=25, value=(3 if i == j else 2), key=f"key_{i}_{j}"
        )

st.success("✅ Key matrix input complete.")
st.write("Key matrix:")
st.write(key_matrix)


st.markdown("---")
st.subheader("✍️ Encrypt / Decrypt Message")

mode = st.radio("Mode", ["Encrypt", "Decrypt"])
text_input = st.text_input("Enter text (A–Z only):", "HELLO")

mod = 26

if st.button("🔁 Run Cipher") and key_matrix is not None:
    try:
        if mode == "Encrypt":
            result = encrypt(text_input, key_matrix, mod)
        else:
            result = decrypt(text_input, key_matrix, mod)
        st.text_area("Result:", value=result, height=100)

        if mode == "Decrypt":
            inv = mod_matrix_inverse(key_matrix, mod)
            st.write("🔁 Inverse Key Matrix mod 26:")
            st.write(inv)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        
st.markdown("---")
st.subheader("🧬 Hill++ Encryption & Decryption (Side-by-Side)")

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### 🔐 Encrypt with Hill++")
    gamma_enc = st.number_input("Gamma (γ) – Encryption", min_value=1, value=3, key="gamma_enc")
    beta_enc_input = st.text_input("Seed β (comma-separated) – Encryption", value="5,12", key="beta_enc")
    text_enc = st.text_input("Enter text to encrypt (A–Z):", "HELLO", key="text_enc")

    if st.button("▶️ Run Encryption"):
        try:
            beta_enc = [int(x.strip()) for x in beta_enc_input.split(",")]
            C_blocks, encrypted_text = hillpp_encrypt(text_enc, key_matrix, gamma_enc, beta_enc)
            st.success(f"Encrypted text: {encrypted_text}")
            st.write("🔐 Cipher blocks:")
            st.write(C_blocks)
        except Exception as e:
            st.error(f"❌ Encryption error: {e}")

with right_col:
    st.markdown("### 🔓 Decrypt with Hill++")
    gamma_dec = st.number_input("Gamma (γ) – Decryption", min_value=1, value=3, key="gamma_dec")
    beta_dec_input = st.text_input("Seed β (comma-separated) – Decryption", value="5,12", key="beta_dec")
    text_dec = st.text_input("Enter text to decrypt (A–Z):", key="text_dec")

    if st.button("▶️ Run Decryption"):
        try:
            beta_dec = [int(x.strip()) for x in beta_dec_input.split(",")]
            P_blocks, decrypted_text = hillpp_decrypt(text_dec, key_matrix, gamma_dec, beta_dec)
            st.success(f"Decrypted text: {decrypted_text}")
            st.write("🔓 Plaintext blocks:")
            st.write(P_blocks)
        except Exception as e:
            st.error(f"❌ Decryption error: {e}")


st.markdown("---")
# UI for selecting generation mode
st.subheader("🔧 Involutory Matrix Generator")
mode = st.radio("Choose method", [
    "Generate 1 involutory matrix",
    "Generate all involutory matrices",
    "Customize using user blocks (A22 and A12)"
])

matrix_size = st.number_input("Matrix size n (nxn):", min_value=2, value=2, step=1)
modulus = st.number_input("Modulus (mod m):", min_value=2, value=26, step=1)

if mode == "Generate 1 involutory matrix":
    if st.button("🔁 Generate One"):
        K = generate_involutory_matrix(matrix_size, modulus)
        if K is not None:
            st.write("✅ Matrix:")
            st.write(K)
            st.write("✅ Verified:", is_involutory(K, modulus))
        else:
            st.error("❌ Could not generate involutory matrix.")

elif mode == "Generate all involutory matrices":
    max_matrices = st.slider("Max matrices to generate", 1, 100, 10)
    if st.button("📚 Generate All"):
        matrices = generate_all_involutory_matrices(matrix_size, modulus, max_matrices)
        st.success(f"✅ {len(matrices)} matrices found.")
        for i, mat in enumerate(matrices):
            st.text(f"Matrix {i+1}:")
            st.write(mat)

elif mode == "Customize using user blocks (A22 and A12)":
    if matrix_size % 2 != 0:
        st.warning("Matrix size must be even for this method.")
    else:
        half_n = matrix_size // 2
        st.markdown("### 🧩 Input A22 block (bottom-right)")
        A22 = np.zeros((half_n, half_n), dtype=int)
        for i in range(half_n):
            cols = st.columns(half_n)
            for j in range(half_n):
                A22[i][j] = cols[j].number_input(f"A22[{i},{j}]", min_value=0, max_value=modulus-1, value=0, key=f"A22_{i}_{j}")

        st.markdown("### 🧩 Input A12 block (top-right)")
        A12 = np.zeros((half_n, half_n), dtype=int)
        for i in range(half_n):
            cols = st.columns(half_n)
            for j in range(half_n):
                A12[i][j] = cols[j].number_input(f"A12[{i},{j}]", min_value=0, max_value=modulus-1, value=1, key=f"A12_{i}_{j}")

        if st.button("🧪 Try Constructing Involutory Matrix"):
            A11 = (-A22) % modulus

            from utils.involutory_finder import construct_from_user_blocks, generate_even_involutory_matrix

            # Try random A21
            K = construct_from_user_blocks(matrix_size, modulus, A22, A12)

            if K is not None:
                st.success("✅ Found valid involutory matrix using your A22 and A12")
                st.write("Matrix K:")
                st.write(K)
            else:
                st.warning("⚠️ Could not find a valid A21 for the given A22 and A12.")

                # Suggest fallback matrix
                fallback = generate_even_involutory_matrix(matrix_size, modulus)
                if fallback is not None:
                    st.markdown("### 🤖 Suggested valid involutory matrix instead:")
                    st.write(fallback)
                    st.write("✅ Verified:", is_involutory(fallback, modulus))
                else:
                    st.error("❌ Fallback suggestion also failed. Try simpler A22/A12.")
