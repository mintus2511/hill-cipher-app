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

st.set_page_config(page_title="🔐 Hill Cipher++", layout="wide")
# ✅ Apply custom CSS to center and limit width
st.markdown("""
    <style>
        .main .block-container {
            max-width: 700px;
            padding-top: 1rem;
            padding-bottom: 1rem;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)
st.title("🔐 Hill Cipher++ Visualization")

mod = 26
block_size = st.number_input("Matrix size (n x n)", min_value=2, max_value=6, value=2, step=1)

# Reset auto_matrix if size changes
if "last_size" not in st.session_state or st.session_state.last_size != block_size:
    st.session_state.auto_matrix = None
    st.session_state.selected_generated_matrix = None
    st.session_state.last_size = block_size

# --- Manual Key Input ---
st.markdown("---")
st.subheader("🔑 Manual Key Matrix Input")
st.markdown(f"Enter your {block_size}×{block_size} key matrix below (mod 26):")

if st.button("🎲 Auto-generate valid involutory matrix"):
    auto = generate_involutory_matrix(block_size, mod)
    if auto is not None:
        st.session_state.auto_matrix = auto
        st.success("✅ Auto-filled a valid involutory matrix!")
    else:
        st.error("❌ Could not generate an involutory matrix.")

# Optional: Select from all generated involutory matrices
with st.expander("📚 Or choose from all generated involutory matrices"):
    max_gen = st.slider("Max matrices to generate", 1, 100, 10)
    if st.button("🔍 Generate All Possible Involutory Matrices"):
        all_matrices = generate_all_involutory_matrices(block_size, mod, max_gen)
        st.session_state.generated_matrices = all_matrices

    if "generated_matrices" in st.session_state:
        matrix_options = {
            f"Matrix {i+1}:\n{np.array2string(m)}": m for i, m in enumerate(st.session_state.generated_matrices)
        }
        selected = st.selectbox("Choose a matrix to use:", list(matrix_options.keys()))
        if selected:
            st.session_state.selected_generated_matrix = matrix_options[selected]
            st.session_state.auto_matrix = st.session_state.selected_generated_matrix
            st.success("✅ Selected matrix applied to key input above.")

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

st.success("✅ Key matrix input complete.")
st.write("Key matrix:")
st.write(key_matrix)

if is_involutory(key_matrix, mod):
    st.success("✅ This key matrix is involutory (K² ≡ I mod 26).")
else:
    st.warning("⚠️ This matrix is not involutory. Hill++ decryption may fail.")

# --- Simple Hill Cipher ---
st.markdown("---")
st.subheader("✍️ Encrypt / Decrypt Message")
mode = st.radio("Mode", ["Encrypt", "Decrypt"])
text_input = st.text_input("Enter text (A–Z only):", "HELLO")

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

# --- Hill++ Cipher ---
st.markdown("---")
st.subheader("🧬 Hill++ Encryption & Decryption (Side-by-Side)")
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### 🔐 Encrypt with Hill++")
    gamma_enc = st.number_input("Gamma (γ) – Encryption", min_value=1, value=3, key="gamma_enc")

    col1, col2 = st.columns([1, 1])
    if col1.button("🎲 Generate β for Encryption"):
        st.session_state.random_beta_enc = list(np.random.randint(0, 26, size=block_size))
    if col2.button("🧹 Reset β (Encryption)"):
        st.session_state.random_beta_enc = None

    st.markdown("#### 🔢 Input Seed β (initial vector) – Encryption")
    beta_enc = []
    cols = st.columns(block_size)
    for i in range(block_size):
        default_val = (
            st.session_state.random_beta_enc[i]
            if st.session_state.random_beta_enc is not None and len(st.session_state.random_beta_enc) == block_size
            else 1
        )
        beta_enc.append(
            cols[i].number_input(f"β[{i}] (enc)", min_value=0, max_value=25, value=default_val, key=f"beta_enc_{i}_left")
        )

    text_enc = st.text_input("Enter text to encrypt (A–Z):", "HELLO", key="text_enc")
    if st.button("▶️ Run Encryption"):
        try:
            C_blocks, encrypted_text = hillpp_encrypt(text_enc, key_matrix, gamma_enc, beta_enc)
            st.success(f"Encrypted text: {encrypted_text}")
            st.write("🔐 Cipher blocks:")
            st.write(C_blocks)
            if st.session_state.use_same_beta:
                st.session_state.random_beta_dec = beta_enc.copy()
        except Exception as e:
            st.error(f"❌ Encryption error: {e}")

with right_col:
    st.markdown("### 🔓 Decrypt with Hill++")
    gamma_dec = st.number_input("Gamma (γ) – Decryption", min_value=1, value=3, key="gamma_dec")

    col_beta, col_info = st.columns([8, 1])
    st.session_state.use_same_beta = col_beta.checkbox("🔁 Use same β from encryption", value=st.session_state.use_same_beta)
    col_info.markdown("ℹ️", help="Automatically copy β (seed vector) from encryption and use it for decryption.")


    if st.session_state.use_same_beta and st.session_state.random_beta_enc:
        st.session_state.random_beta_dec = st.session_state.random_beta_enc.copy()

    col3, col4 = st.columns([1, 1])
    if col3.button("🎲 Generate β for Decryption"):
        st.session_state.random_beta_dec = list(np.random.randint(0, 26, size=block_size))
    if col4.button("🧹 Reset β (Decryption)"):
        st.session_state.random_beta_dec = None

    st.markdown("#### 🔢 Input Seed β (initial vector) – Decryption")
    beta_dec = []
    cols = st.columns(block_size)
    for i in range(block_size):
        default_val = (
            st.session_state.random_beta_dec[i]
            if st.session_state.random_beta_dec is not None and len(st.session_state.random_beta_dec) == block_size
            else 1
        )
        beta_dec.append(
            cols[i].number_input(f"β[{i}] (dec)", min_value=0, max_value=25, value=default_val, key=f"beta_dec_{i}_right")
        )

    text_dec = st.text_input("Enter text to decrypt (A–Z):", key="text_dec")
    if st.button("▶️ Run Decryption"):
        try:
            P_blocks, decrypted_text = hillpp_decrypt(text_dec, key_matrix, gamma_dec, beta_dec)
            st.success(f"Decrypted text: {decrypted_text}")
            st.write("🔓 Plaintext blocks:")
            st.write(P_blocks)
        except Exception as e:
            st.error(f"❌ Decryption error: {e}")
