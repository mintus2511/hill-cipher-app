
import streamlit as st
import numpy as np
from cipher_logic import encrypt, decrypt, mod_matrix_inverse
from utils.involutory_finder import (
    generate_involutory_matrix,
    generate_all_involutory_matrices,
    construct_from_user_blocks,
    is_involutory
)

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

key_input = st.text_area(f"Enter your {block_size}x{block_size} key matrix (comma-separated rows):", 
                         value="3,3\n2,5" if block_size==2 else "6,24,1\n13,16,10\n20,17,15")

try:
    key_matrix = np.array([[int(num) for num in row.split(",")] for row in key_input.strip().split("\n")])
    assert key_matrix.shape == (block_size, block_size)
    st.success("✅ Matrix loaded successfully")
except:
    st.error("⚠️ Invalid matrix format.")
    key_matrix = None

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
        st.markdown("### 🔢 Enter A22 matrix (space-separated values per row):")
        A22_rows = [st.text_input(f"A22 Row {i+1}", value=" ".join(["0"] * half_n)) for i in range(half_n)]

        st.markdown("### 🔢 Enter A12 matrix (space-separated values per row):")
        A12_rows = [st.text_input(f"A12 Row {i+1}", value=" ".join(["1"] * half_n)) for i in range(half_n)]

        if st.button("🧪 Generate from A22 & A12"):
            A22_mat = parse_text_matrix(A22_rows, modulus)
            A12_mat = parse_text_matrix(A12_rows, modulus)

            if A22_mat is None or A12_mat is None:
                st.error("❌ Invalid matrix input. Please enter only space-separated integers.")
            else:
                try:
                    from utils.involutory_finder import construct_from_user_blocks
                    K = construct_from_user_blocks(matrix_size, modulus, A22_mat, A12_mat)
                    if K is not None:
                        st.write("✅ Involutory Matrix:")
                        st.write(K)
                        st.write("✅ Verified:", is_involutory(K, modulus))
                    else:
                        st.warning("⚠️ No valid A21 matrix found after 1000 attempts.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
