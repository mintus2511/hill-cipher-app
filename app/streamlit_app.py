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

# ====== STATE INIT ======
if "random_beta" not in st.session_state:
    st.session_state.random_beta = None

if "auto_matrix" not in st.session_state:
    st.session_state.auto_matrix = None

if "sync_beta" not in st.session_state:
    st.session_state.sync_beta = False

# ====== UI CONFIG ======
st.set_page_config(page_title="🔐 Hill++ Cipher App", layout="wide")
st.title("🔐 Hill++ Cipher Visualization")

# ====== SHARED PARAMETERS ======
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    mod = st.number_input("Modulus (m):", min_value=2, value=26, step=1)
with col2:
    gamma = st.number_input("Gamma (secret multiplier)", min_value=1, value=3)
with col3:
    block_size = st.number_input("Matrix size (n x n):", min_value=2, max_value=6, value=2, step=1)

# ====== SEED VECTOR ======
st.markdown("### 🔢 Seed vector \u03b2 (length n)")
beta_cols = st.columns(block_size)
if st.button("🎲 Generate random \u03b2"):
    st.session_state.random_beta = list(np.random.randint(0, mod, size=block_size))

beta = []
for i in range(block_size):
    default_val = (
        st.session_state.random_beta[i]
        if st.session_state.random_beta and len(st.session_state.random_beta) == block_size
        else 1
    )
    beta.append(beta_cols[i].number_input(f"\u03b2[{i}]", min_value=0, max_value=mod-1, value=default_val, key=f"beta_{i}"))

# ====== INVOLUTORY MATRIX INPUT ======
st.markdown("### 🔐 Involutory matrix (n x n)")
if st.button("🎲 Auto-generate involutory matrix"):
    generated = generate_involutory_matrix(block_size, mod)
    if generated is not None:
        st.session_state.auto_matrix = generated
        st.success("✅ Generated valid involutory matrix")
    else:
        st.error("❌ Generation failed. Try smaller size or mod.")

key_matrix = np.zeros((block_size, block_size), dtype=int)
for i in range(block_size):
    row_cols = st.columns(block_size)
    for j in range(block_size):
        default_val = (
            int(st.session_state.auto_matrix[i][j])
            if st.session_state.auto_matrix is not None and st.session_state.auto_matrix.shape == (block_size, block_size)
            else (3 if i == j else 2)
        )
        key_matrix[i][j] = row_cols[j].number_input(
            f"Key[{i},{j}]", min_value=0, max_value=mod-1, value=default_val, key=f"key_{i}_{j}"
        )

if is_involutory(key_matrix, mod):
    st.success("✅ This key matrix is involutory (K² ≡ I mod m)")
else:
    st.warning("⚠️ Matrix is NOT involutory. Hill++ decryption may fail")

# ====== TEXT INPUT ======
st.markdown("### 📝 Ciphertext / Plaintext Input")
text_input = st.text_input("Enter your message (A-Z only):", "HELLO")

# ====== ENCRYPT / DECRYPT SIDE BY SIDE ======
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔐 Encrypt")
    if st.button("▶️ Run Hill++ Encryption"):
        try:
            C_blocks, encrypted_text = hillpp_encrypt(text_input, key_matrix, gamma, beta)
            st.success(f"Encrypted text: {encrypted_text}")
            st.write("🔐 Cipher blocks:")
            st.write(C_blocks)
            if st.checkbox("🔁 Use this \u03b2 for decryption?"):
                st.session_state.sync_beta = True
                st.session_state.random_beta = beta.copy()
        except Exception as e:
            st.error(f"❌ Encryption error: {e}")

with col_right:
    st.subheader("🔓 Decrypt")
    if st.session_state.sync_beta:
        st.info("ℹ️ Using \u03b2 from encryption")
        beta_dec = st.session_state.random_beta
    else:
        beta_dec = []
        beta_dec_cols = st.columns(block_size)
        for i in range(block_size):
            beta_dec.append(
                beta_dec_cols[i].number_input(
                    f"\u03b2[{i}] (decrypt)", min_value=0, max_value=mod-1, value=1, key=f"beta_dec_{i}"
                )
            )

    text_decrypt = st.text_input("Enter text to decrypt (A-Z):", "", key="text_decrypt")

    if st.button("▶️ Run Hill++ Decryption"):
        try:
            P_blocks, decrypted_text = hillpp_decrypt(text_decrypt, key_matrix, gamma, beta_dec)
            st.success(f"Decrypted text: {decrypted_text}")
            st.write("🔓 Plaintext blocks:")
            st.write(P_blocks)
        except Exception as e:
            st.error(f"❌ Decryption error: {e}")
