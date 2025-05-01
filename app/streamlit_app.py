
import streamlit as st
import numpy as np
from cipher_logic import encrypt, decrypt, mod_matrix_inverse
import sys
import os
sys.path.append(os.path.abspath(".."))  # So Python can find parent folders

from utils.involutory_finder import find_involutory_matrices

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
st.subheader("🧠 Auto-Generate Involutory Key Matrices")

if block_size != 2:
    st.warning("⚠️ Only supported for 2x2 matrices due to performance limits.")
else:
    if st.checkbox("🔍 Show all 2x2 involutory matrices (A² ≡ I mod 26)"):
        with st.spinner("Calculating..."):
            matrices = find_involutory_matrices(2, 26)

        st.success(f"✅ Found {len(matrices)} involutory matrices mod 26")
        for i, mat in enumerate(matrices):
            st.text(f"Matrix {i+1}:\n{mat}")
