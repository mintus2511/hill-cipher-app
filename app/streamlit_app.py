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
if "use_same_gamma" not in st.session_state:
    st.session_state.use_same_gamma = False
if "selected_generated_matrix" not in st.session_state:
    st.session_state.selected_generated_matrix = None
if "section" not in st.session_state:
    st.session_state.section = "Hill Cipher"
if "matrix_mode" not in st.session_state:
    st.session_state.matrix_mode = "Manual"

st.set_page_config(page_title="🔐 Hill Cipher", layout="centered")
st.title("""🔐 **Hill Cipher/Hill++ Visualization**""")

# --- Navigation ---
st.session_state.section = st.radio(
    "🔍 Choose a section:",
    ["User Guide", "Hill Cipher", "Hill++"]
)

if st.session_state.section == "User Guide":
    st.markdown("""
    ## 📘 User Guide
    
    **Hill Cipher:**
    
    ***What is Hill Cipher?***
    - Hill Cipher was invented by American mathematician Lester S. Hill in 1929, the Hill cipher marked a significant advance in classical cryptography. It was the first practical polygraphic substitution cipher that could encrypt blocks of more than three letters at a time, making it a true block cipher. 
    
    ***Input Requirement*** 
    - Plaintext: Enter the message you want to encrypt. Only alphabetic characters are processed; spaces and punctuation should be removed or handled separately. You must write UPPERCASE letters.
    - Block size (n): The dimension of the key matrix (e.g., 2 or 3). The plaintext will be split into blocks of this size. You must write UPPERCASE letters.
    - Key matrix: Users can enter the matrix manually, auto-generated, or choose from the list of available matrices. One requirement for key matrices is that they have to be invertible.
    
    ***How to Use***
    1. Enter your plaintext message.
    2. Specify the block size (n), and key matrix.
    3. Click **Encrypt** to generate the ciphertext.
    4. To decrypt, enter the ciphertext and the same parameters, then click **Decrypt**.


    **Hill ++** 

    ***What is Hill++?***
    - Hill++ is an enhanced version of the classical Hill cipher that improves security by using an involutory key matrix (a matrix that is its own inverse) and a dynamic random matrix key (RMK) generated from previous ciphertext blocks. This approach eliminates the need for computing matrix inverses during decryption and increases resistance to known-plaintext attacks.
    
    ***Input Requirements***
    - Plaintext: Enter the message you want to encrypt. Only alphabetic characters are processed; spaces and punctuation should be removed or handled separately.
    - Block size (n): The dimension of the key matrix (e.g., 2 or 3). The plaintext will be split into blocks of this size.
    - Seed vector (β): An initial numeric vector of length n used as the starting random matrix key (RMK).
    - Multiplying factor (γ): A numeric factor used to generate RMK for subsequent blocks from previous ciphertext blocks.
    - Modulus (m): The modulus for arithmetic operations, typically 26 for the English alphabet.
    
    ***How to Use***
    1. Enter your plaintext message.
    2. Specify the block size (n), seed vector (β), multiplying factor (γ), and modulus (m).
    3. Click **Encrypt** to generate the ciphertext.
    4. To decrypt, enter the ciphertext and the same parameters, then click **Decrypt**.
    
    ***Note***
     
    - Not all matrices are valid keys. The key matrix must be a square involutory matrix modulo m, meaning it satisfies:
    **K^2≡I (modm)**
    where I is the identity matrix.  If these conditions are not met, the matrix cannot be used as a key.
    """)

def pad_text(text, block_size, filler='X'):
    text = ''.join(filter(str.isalpha, text.upper()))
    return text + filler * ((block_size - len(text) % block_size) % block_size)

def strip_padding(text, filler='X'):
    return text.rstrip(filler)

if st.session_state.section in ["Hill Cipher", "Hill++"]:
    mod = 26
    block_size = st.number_input(
        "Matrix size (n x n)",
        min_value=2, max_value=6, value=2, step=1,
        help="The dimension of the key matrix (e.g., 2 or 3). The plaintext will be split into blocks of this size."
    )

    if "last_size" not in st.session_state or st.session_state.last_size != block_size:
        st.session_state.auto_matrix = None
        st.session_state.selected_generated_matrix = None
        st.session_state.last_size = block_size

    st.markdown("---")
    st.subheader("🔑 Key Matrix Setup")

    st.session_state.matrix_mode = st.radio(
        "Choose matrix input mode:",
        ["Manual", "Auto-generate", "Generate all possible key matrix"],
        key="matrix_mode_selector",
        help="""**Key matrix:** Users can enter the matrix manually, auto-generated, or choose from the list of available matrices. One requirement for key matrices is that they have to be invertible (for Hill Cipher) or involuntary (for Hill++)."""
    )

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
                    f"Key[{i},{j}]", min_value=0, max_value=25, value=default_val, key=f"key_{i}_{j}",
                    help="Manually input each element of the key matrix"
                )
        st.session_state.key_matrix = key_matrix
        st.write("Key matrix:")
        st.write(key_matrix)

    elif st.session_state.matrix_mode == "Auto-generate":
        generate_button_label = "🎲 Generate Involutory Matrix" if st.session_state.section == "Hill++" else "🎲 Generate Invertible Matrix"
        if st.button(generate_button_label, help="Automatically create a valid key matrix for calculation."):
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

    elif st.session_state.matrix_mode == "Generate all possible key matrix":
        max_gen = st.slider(
            "Max matrices to generate", 1, 100, 10,
            help="Choose how many candidate matrices to generate and pick one to use"
        )
        if st.button("🔍 Generate All Matrices", help="Generate a list of all valid matrices to choose from"):
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
            matrix_options = {f"Matrix {i+1}": m for i, m in enumerate(st.session_state.generated_matrices)}
            selected = st.selectbox(
                "Choose a matrix to use:",
                list(matrix_options.keys()),
                help="Pick one matrix from the generated list to use"
            )
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
    text_input = st.text_input("Enter plaintext (A–Z only):", "HELLO",
                              help="""**Plaintext:** Enter the message you want to encrypt. Only alphabetic characters are processed; spaces and punctuation should be removed or handled separately.
                              
                              **Ciphertext:** Only uppercase English letters (A–Z). No spaces or special characters.""")

    if st.button("🔁 Run Cipher"):
        try:
            filtered_text = ''.join(filter(str.isalpha, text_input.upper()))
            if not filtered_text:
                raise ValueError("Input must contain at least one valid A–Z character (A–Z only).")
            
            padded_text = pad_text(filtered_text, st.session_state.key_matrix.shape[0], filler='X')
            
            if mode == "Encrypt":
                result = encrypt(padded_text, st.session_state.key_matrix, mod)
                st.text_area("Result:", value=strip_padding(result), height=100)
    
                if show_steps:
                    st.write("### 🔎 Step-by-step Encryption")
                    st.write("1. Preprocessed text:", filtered_text)
                    numeric = text_to_numbers(''.join(filter(str.isalpha, padded_text.upper())))
                    st.write("2. Numeric form:", numeric)
                    st.write("3. Multiply each block by key matrix and take mod 26")
    
            else:
                result = decrypt(text_input, st.session_state.key_matrix, mod)
                st.text_area("Result:", value=strip_padding(result), height=100)
    
                if show_steps:
                    st.write("### 🔎 Step-by-step Decryption")
                    numeric = text_to_numbers(''.join(filter(str.isalpha, text_input.upper())))
                    st.write("1. Numeric form:", numeric)
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

        text_enc = st.text_input("Enter plaintext to encrypt (A–Z only):", "HELLO", key="hillpp_text_enc",
                                help="""**Plaintext:**
                                        - Only uppercase English letters (A–Z).
                                        - No spaces, numbers, or special characters.
                                        - Length should be a multiple of block size (padding with 'X' is automatic if needed).
                                        """)
        gamma = st.number_input("Gamma (γ)", min_value=1, value=3, key="gamma_enc",
                               help="""**Secret Multiplier:** Integer between 0 and m-1 (usually 0-25) """)

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
                cols[i].number_input(f"β[{i}]", min_value=0, max_value=25, value=default_val, key=f"beta_{i}_hillpp",
                                    help="Manually input each element of the seed vector β")
            )

        st.session_state.use_same_beta = st.checkbox("🔁 Use same β for decryption", help="Use the same β (seed vector) from encryption for decryption")
        if st.session_state.use_same_beta:
            st.session_state.random_beta_dec = beta_enc.copy()

        if st.button("▶️ Encrypt (Hill++)"):
            try:
                if show_steps:
                    steps, result, original_length = hillpp_encrypt_verbose(text_enc, st.session_state.key_matrix, gamma, beta_enc)
                    st.session_state.original_length = original_length
                    st.success(f"Encrypted text: {result}")
                    st.write("### 🧮 Steps")
                    for s in steps:
                        st.write(s)
                else:
                    C_blocks, encrypted_text, original_length = hillpp_encrypt(text_enc, st.session_state.key_matrix, gamma, beta_enc)
                    st.session_state.original_length = original_length
                    st.success(f"Encrypted text: {encrypted_text}")
                    st.write("🔐 Cipher blocks:")
                    st.write(C_blocks)
            except Exception as e:
                st.error(f"❌ Encryption error: {e}")

    with right_col:
        st.markdown("### 🔓 Decrypt")
        text_dec = st.text_input("Enter plaintext to decrypt (A–Z only):", key="hillpp_text_dec",
                                help="""**Ciphertext:**
                                        - Only uppercase English letters (A–Z).
                                        - No spaces or special characters.
                                        """)

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
                cols[i].number_input(f"β[{i}] (dec)", min_value=0, max_value=25, value=default_val, key=f"beta_{i}_dec_hillpp",
                                    help="Manually input each element of the seed vector β")
            )

        st.session_state.use_same_gamma = st.checkbox("🔁 Use same γ as encryption", value=False, help="Use the same γ multiplier value as in encryption")
        if st.session_state.use_same_gamma:
            gamma_dec = st.session_state.get("gamma_enc", 3)
        else:
            gamma_dec = st.number_input("Gamma (γ) for decryption", min_value=1, value=3, key="gamma_dec",
                                       help="""**Secret Multiplier:** Integer between 0 and m-1 (usually 0-25) """)

        if st.button("▶️ Decrypt (Hill++)"):
            try:
                original_length = st.session_state.get("original_length", len(text_dec))
                if show_steps:
                    steps, result = hillpp_decrypt_verbose(text_dec, st.session_state.key_matrix, gamma_dec, beta_dec, original_length)
                    st.success(f"Decrypted text: {result}")
                    st.write("### 🧮 Steps")
                    for s in steps:
                        st.write(s)
                else:
                    P_blocks, decrypted_text = hillpp_decrypt(text_dec, st.session_state.key_matrix, gamma_dec, beta_dec, original_length)
                    st.success(f"Decrypted text: {decrypted_text}")
                    st.write("🔓 Plaintext blocks:")
                    st.write(P_blocks)
            except Exception as e:
                st.error(f"❌ Decryption error: {e}")
