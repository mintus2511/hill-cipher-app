import streamlit as st
import numpy as np
from cipher_logic import encrypt, decrypt, mod_matrix_inverse, is_invertible_matrix, text_to_numbers, numbers_to_text
from utils.involutory_finder import (
    generate_involutory_matrix,
    generate_all_involutory_matrices,
    is_involutory
)
from utils.hillpp_cipher import hillpp_encrypt, hillpp_decrypt, hillpp_encrypt_verbose, hillpp_decrypt_verbose

# Session state setup
if "random_beta_enc" not in st.session_state:
    st.session_state.random_beta_enc = None
if "random_beta_dec" not in st.session_state:
    st.session_state.random_beta_dec = None
if "auto_matrix" not in st.session_state:
    st.session_state.auto_matrix = None
if "use_same_beta" not in st.session_state:
    st.session_state.use_same_beta = False
if "selected_generated_matrix" not in st.session_state:
    st.session_state.selected_generated_matrix = None
if "section" not in st.session_state:
    st.session_state.section = "Hill Cipher"
if "matrix_mode" not in st.session_state:
    st.session_state.matrix_mode = "Manual"

st.set_page_config(page_title="🔐 Hill Cipher", layout="centered")
st.title("🔐 Hill Cipher Visualization")

# --- Navigation ---
st.session_state.section = st.radio(
    "🔍 Choose a section:",
    ["User Guide", "Hill Cipher", "Hill++"]
)

if st.session_state.section == "User Guide":
    st.markdown("""
    ## 📘 User Guide

    **1. What is Hill Cipher?**
    - A secure variant of the Hill cipher using involutory or invertible matrices.

    **2. How to use:**
    - Choose matrix size (2–6)
    - Manually input or auto-generate a key matrix
    - Enter plaintext or ciphertext (A–Z only)
    - For Hill++: specify gamma (γ) and initial vector β

    **3. Notes:**
    - Use uppercase English letters only (A–Z)
    - The key matrix must be involutory (K² ≡ I mod 26) or invertible for Hill Cipher
    - Hill++ decryption requires correct β and γ values

    👉 Use the top menu to navigate to encryption sections.
    """)

def pad_text(text, block_size, filler='X'):
    text = ''.join(filter(str.isalpha, text.upper()))
    return text + filler * ((block_size - len(text) % block_size) % block_size)

def strip_padding(text, filler='X'):
    return text.rstrip(filler)

def decrypt_strip(text, key_matrix, mod=26, filler='X'):
    result = decrypt(text, key_matrix, mod)
    return strip_padding(result, filler)

def hillpp_decrypt_strip(text, key_matrix, gamma, beta, mod=26, filler='X'):
    _, result = hillpp_decrypt(text, key_matrix, gamma, beta, mod)
    return strip_padding(result, filler)

def hillpp_decrypt_verbose_strip(text, key_matrix, gamma, beta, mod=26, filler='X'):
    steps, result = hillpp_decrypt_verbose(text, key_matrix, gamma, beta, mod)
    return steps, strip_padding(result, filler)

if st.session_state.section in ["Hill Cipher", "Hill++"]:
    mod = 26
    block_size = st.number_input("Matrix size (n x n)", min_value=2, max_value=6, value=2, step=1)

    if "last_size" not in st.session_state or st.session_state.last_size != block_size:
        st.session_state.auto_matrix = None
        st.session_state.selected_generated_matrix = None
        st.session_state.last_size = block_size

    st.markdown("---")
    st.subheader("🔑 Key Matrix Setup")

    st.session_state.matrix_mode = st.radio("Choose matrix input mode:", ["Manual", "Auto-generate", "Choose from list"], key="matrix_mode_selector")

    if st.session_state.matrix_mode == "Manual":
        st.markdown(f"Enter your {block_size}×{block_size} key matrix below (mod 26):")
        key_matrix = np.zeros((block_size, block_size), dtype=int)
        for i in range(block_size):
            cols = st.columns(block_size)
            for j in range(block_size):
                default_val = (
                    int(st.session_state.auto_matrix[i][j])
                    if st.session_state.auto_matrix is not None and st.session_state.auto_matrix.shape == (block_size, block_size)
                    else (3 if i == j else 2)
                )
                key_matrix[i][j] = cols[j].number_input(
                    f"Key[{i},{j}]", min_value=0, max_value=25, value=default_val, key=f"key_{i}_{j}"
                )
        st.session_state.key_matrix = key_matrix
        st.write("Key matrix:")
        st.write(key_matrix)

    elif st.session_state.matrix_mode == "Auto-generate":
        generate_button_label = "🎲 Generate Involutory Matrix" if st.session_state.section == "Hill++" else "🎲 Generate Invertible Matrix"
        if st.button(generate_button_label):
            if st.session_state.section == "Hill++":
                matrices = generate_all_involutory_matrices(block_size, mod, max_matrices=1)
                if matrices:
                    st.session_state.key_matrix = matrices[0]
                    st.success("✅ Involutory matrix generated!")
                    st.write(matrices[0])
                else:
                    st.error("❌ Failed to generate involutory matrix.")
            else:
                found = False
                attempts = 0
                while not found and attempts < 100:
                    random_matrix = np.random.randint(0, 26, size=(block_size, block_size))
                    if is_invertible_matrix(random_matrix, mod):
                        st.session_state.key_matrix = random_matrix
                        st.success("✅ Invertible matrix generated!")
                        st.write(random_matrix)
                        found = True
                    attempts += 1
                if not found:
                    st.error("❌ Could not generate a valid matrix after 100 attempts.")

    elif st.session_state.matrix_mode == "Choose from list":
        max_gen = st.slider("Max matrices to generate", 1, 100, 10)
        if st.button("🔍 Generate All Matrices"):
            if st.session_state.section == "Hill++":
                matrices = generate_all_involutory_matrices(block_size, mod, max_gen)
            else:
                matrices = []
                attempts = 0
                while len(matrices) < max_gen and attempts < max_gen * 10:
                    candidate = np.random.randint(0, 26, size=(block_size, block_size))
                    if is_invertible_matrix(candidate, mod):
                        matrices.append(candidate)
                    attempts += 1
            st.session_state.generated_matrices = matrices

        if "generated_matrices" in st.session_state:
            matrix_options = {f"Matrix {i+1}:": m for i, m in enumerate(st.session_state.generated_matrices)}
            selected = st.selectbox("Choose a matrix to use:", list(matrix_options.keys()))
            if selected:
                st.session_state.key_matrix = matrix_options[selected]
                st.success("✅ Matrix selected and applied.")
                st.code(np.array2string(matrix_options[selected]), language="text")

    if "key_matrix" in st.session_state:
        if st.session_state.section == "Hill Cipher":
            if is_invertible_matrix(st.session_state.key_matrix, mod):
                st.success("✅ Key matrix is invertible.")
            else:
                st.warning("⚠️ This matrix is not invertible mod 26.")
        else:
            if is_involutory(st.session_state.key_matrix, mod):
                st.success("✅ Matrix is involutory (K² ≡ I mod 26).")
            else:
                st.warning("⚠️ Matrix is not involutory.")

# Step-by-step option
show_steps = st.checkbox("📘 Show step-by-step calculation")

if st.session_state.section == "Hill Cipher":
    st.markdown("---")
    st.subheader("✍️ Encrypt / Decrypt Message")
    mode = st.radio("Mode", ["Encrypt", "Decrypt"])
    text_input = st.text_input("Enter text (A–Z only):", "HELLO")

    if st.button("🔁 Run Cipher"):
        try:
            padded_text = pad_text(text_input, st.session_state.key_matrix.shape[0], filler='/')
            if mode == "Encrypt":
                result = encrypt(padded_text, st.session_state.key_matrix, mod)
                st.text_area("Result:", value=result, height=100)

                if show_steps:
                    st.write("### 🔎 Step-by-step Encryption")
                    st.write("1. Preprocessed text:", padded_text)
                    st.write("2. Numeric form:", text_to_numbers(padded_text))
                    st.write("3. Multiply each block by key matrix and take mod 26")

            else:
                result = decrypt(text_input, st.session_state.key_matrix, mod)
                st.text_area("Result:", value=result, height=100)

                if show_steps:
                    st.write("### 🔎 Step-by-step Decryption")
                    st.write("1. Numeric form:", text_to_numbers(text_input))
                    inv = mod_matrix_inverse(st.session_state.key_matrix, mod)
                    st.write("2. Inverse matrix:")
                    st.write(inv)
                    st.write("3. Multiply each block by inverse matrix and take mod 26")

        except Exception as e:
            st.error(f"❌ Error: {e}")

if st.session_state.section == "Hill++":
    st.markdown("## 🔐 Hill++ Mode")
    mod = 26
    block_size = st.session_state.last_size if "last_size" in st.session_state else 2

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### 🔐 Encrypt")
        gamma = st.number_input("Gamma (γ)", min_value=1, value=3, key="gamma_enc")

        col1, col2 = st.columns([1, 1])
        if col1.button("🎲 Generate β (Seed Vector)"):
            st.session_state.random_beta_enc = list(np.random.randint(0, 26, size=block_size))
        if col2.button("🧹 Reset β"):
            st.session_state.random_beta_enc = None

        st.markdown("#### 🔢 Enter Seed Vector β")
        beta_enc = []
        cols = st.columns(block_size)
        for i in range(block_size):
            default_val = (
                st.session_state.random_beta_enc[i]
                if st.session_state.random_beta_enc is not None and len(st.session_state.random_beta_enc) == block_size
                else 1
            )
            beta_enc.append(
                cols[i].number_input(f"β[{i}]", min_value=0, max_value=25, value=default_val, key=f"beta_{i}_hillpp")
            )

        st.session_state.use_same_beta = st.checkbox("🔁 Use same β for decryption")
        if st.session_state.use_same_beta:
            st.session_state.random_beta_dec = beta_enc.copy()

        text_enc = st.text_input("Enter text to encrypt (A–Z):", "HELLO", key="hillpp_text_enc")
        if st.button("▶️ Encrypt (Hill++)"):
            try:
                if show_steps:
                    steps, result = hillpp_encrypt_verbose(text_enc, st.session_state.key_matrix, gamma, beta_enc)
                    st.success(f"Encrypted text: {result}")
                    st.write("### 🧮 Steps")
                    for s in steps:
                        st.write(s)
                else:
                    C_blocks, encrypted_text = hillpp_encrypt(text_enc, st.session_state.key_matrix, gamma, beta_enc)
                    st.success(f"Encrypted text: {encrypted_text}")
                    st.write("🔐 Cipher blocks:")
                    st.write(C_blocks)
            except Exception as e:
                st.error(f"❌ Encryption error: {e}")

    with right_col:
        st.markdown("### 🔓 Decrypt")
        text_dec = st.text_input("Enter text to decrypt (A–Z):", key="hillpp_text_dec")

        st.markdown("#### 🔢 Enter β for decryption")
        beta_dec = []
        cols = st.columns(block_size)
        for i in range(block_size):
            default_val = (
                st.session_state.random_beta_dec[i]
                if st.session_state.random_beta_dec is not None and len(st.session_state.random_beta_dec) == block_size
                else 1
            )
            beta_dec.append(
                cols[i].number_input(f"β[{i}] (dec)", min_value=0, max_value=25, value=default_val, key=f"beta_{i}_dec_hillpp")
            )

        if st.button("▶️ Decrypt (Hill++)"):
            try:
                if show_steps:
                    steps, result = hillpp_decrypt_verbose(text_dec, st.session_state.key_matrix, gamma, beta_dec)
                    st.success(f"Decrypted text: {result}")
                    st.write("### 🧮 Steps")
                    for s in steps:
                        st.write(s)
                else:
                    P_blocks, decrypted_text = hillpp_decrypt(text_dec, st.session_state.key_matrix, gamma, beta_dec)
                    st.success(f"Decrypted text: {decrypted_text}")
                    st.write("🔓 Plaintext blocks:")
                    st.write(P_blocks)
            except Exception as e:
                st.error(f"❌ Decryption error: {e}")
