
import streamlit as st
import numpy as np
from cipher_logic import encrypt, decrypt, mod_matrix_inverse
from utils.involutory_finder import generate_involutory_matrix

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
st.subheader("🧠 Generate Involutory Key Matrix")

with st.expander("🔧 Matrix Generator Settings"):
    inv_size = st.number_input("Matrix size (n x n):", min_value=2, max_value=8, value=2)
    inv_mod = st.number_input("Modulus (mod N):", min_value=2, value=26)
    if st.button("🔁 Generate Involutory Matrix"):
        try:
            inv_matrix = generate_involutory_matrix(inv_size, inv_mod)
            st.write("✅ Involutory Matrix Found:")
            st.write(inv_matrix)
        except Exception as e:
            st.error(f"❌ Failed to generate matrix: {e}")
