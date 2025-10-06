import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Tokenizer ---
tokenizer_path = "tokenizer.pickle"  # Ensure this path is correct
with open(tokenizer_path, "rb") as handle:
    tokenizer = pickle.load(handle)

# --- Load Model (cached and legacy-safe) ---
@st.cache_resource
def load_model_once():
    try:
        # Try normal load (for newer Keras models)
        model = tf.keras.models.load_model("sarcasm_model.h5", compile=False)
    except Exception as e:
        st.warning(f"Normal load failed ({e}). Trying legacy loader...")
        try:
            from keras.src.legacy.saving import legacy_h5_format
            model = legacy_h5_format.load_model_from_hdf5("sarcasm_model.h5")
        except Exception as e2:
            st.error(f"Legacy load also failed: {e2}")
            raise e2
    return model

model = load_model_once()

# --- Define maxlen (must match training) ---
maxlen = 240  # Replace with actual training maxlen if different

# --- Preprocessing functions ---
def cleanData(text):
    text = re.sub(r"\d+", "", text)
    text = "".join([char for char in text if char not in string.punctuation])
    return text

def preprocess_text(text):
    cleaned_text = cleanData(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    return padded_sequence

# --- Prediction function ---
def predict_sarcasm(headline):
    processed_headline = preprocess_text(headline)
    # Make sure eager execution is enabled for prediction
    tf.config.run_functions_eagerly(True)
    prediction = model.predict(processed_headline)
    tf.config.run_functions_eagerly(False)
    # Handle sigmoid (1 neuron) or softmax (2 neurons)
    if prediction.shape[1] == 1:
        predicted_class = int(prediction[0][0] > 0.5)
    else:
        predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# --- Streamlit App Interface ---
st.title("Sarcasm Detection App")
st.write("Enter a headline to check if it's sarcastic or not.")

headline_input = st.text_area("Enter Headline:")

if st.button("Detect Sarcasm"):
    if headline_input:
        result = predict_sarcasm(headline_input)
        if result == 1:
            st.success("This headline is likely **sarcastic**.")
        else:
            st.info("This headline is likely **not sarcastic**.")
    else:
        st.warning("Please enter a headline to analyze.")