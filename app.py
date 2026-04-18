import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load resources (cached)
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    # Reverse mapping (FAST lookup)
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    return model, tokenizer, max_len, index_to_word


model, tokenizer, max_len, index_to_word = load_resources()


# ------------------------------
# Prediction logic
# ------------------------------
def predict_next_word(text, temperature=1.0):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

    preds = model.predict(sequence, verbose=0)[0]

    # Temperature scaling (for better randomness)
    preds = np.log(preds + 1e-8) / temperature
    preds = np.exp(preds) / np.sum(np.exp(preds))

    predicted_index = np.random.choice(len(preds), p=preds)

    return index_to_word.get(predicted_index, "")


def generate_text(seed_text, n_words, temperature):
    output = seed_text

    for _ in range(n_words):
        next_word = predict_next_word(output, temperature)

        if next_word == "":
            break

        output += " " + next_word

    return output


# ------------------------------
# UI DESIGN
# ------------------------------
st.set_page_config(
    page_title="Next Word Generator",
    page_icon="🧠",
    layout="centered"
)

# Header
st.markdown(
    """
    <h1 style='text-align: center;'>🧠 AI Text Generator</h1>
    <p style='text-align: center; color: gray;'>
    LSTM-based Next Word Prediction | Built with TensorFlow + Streamlit
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Input Section
st.subheader("✍️ Enter your text")

user_input = st.text_area(
    "",
    placeholder="Start typing... (e.g., 'are you a')",
    height=100
)

# Controls
col1, col2 = st.columns(2)

with col1:
    num_words = st.slider("🔢 Words to generate", 1, 30, 10)

with col2:
    temperature = st.slider("🔥 Creativity", 0.5, 1.5, 1.0)

st.markdown("---")

# Button
if st.button("🚀 Generate Text"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Generating..."):
            result = generate_text(user_input.lower(), num_words, temperature)

        st.success("✅ Generated Text:")

        st.markdown(
            f"""
            <div style="
                background-color:#1f2937;
                color:#ffffff;
                padding:18px;
                border-radius:12px;
                font-size:18px;
                line-height:1.6;
                box-shadow:0px 4px 12px rgba(0,0,0,0.3);
            ">
            {result}
            </div>
            """,
            unsafe_allow_html=True
        )


# ------------------------------
# Footer
# ------------------------------
st.markdown("---")

st.markdown(
    """
    <center>
    <b>Tech Stack:</b> TensorFlow | LSTM | Streamlit  
    <br>
    Built as a mini NLP project 🚀
    </center>
    """,
    unsafe_allow_html=True
)