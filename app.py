import streamlit as st
import pandas as pd
import numpy as np
import string
import os

from deep_translator import GoogleTranslator
from gtts import gTTS

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import tensorflow.keras.utils as ku

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Predictive Text Generator",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 Language Selection")

language = st.sidebar.selectbox(
    "Select Output Language",
    ["English", "Hindi", "Telugu", "Tamil"]
)

lang_codes = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta"
}

# ---------------- TITLE ----------------
st.title("🧠 AI-Powered Predictive Text Generator")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Text_Generation_831_rows.csv")
    return df

df = load_data()

# ---------------- CLEAN TEXT ----------------
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii", 'ignore')
    return txt

corpus = [clean_text(x) for x in df['headline'].dropna().values]

# ---------------- TOKENIZER ----------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1

# ---------------- CREATE SEQUENCES ----------------
input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# ---------------- PAD SEQUENCES ----------------
max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding='pre'
    )
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = ku.to_categorical(y, num_classes=total_words)

# ---------------- MODEL ----------------
@st.cache_resource
def train_model():

    model = Sequential()

    model.add(
        Embedding(
            total_words,
            100,
            input_length=max_sequence_len - 1
        )
    )

    # IMPROVED LSTM
    model.add(LSTM(200))

    model.add(Dense(total_words, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # IMPROVED EPOCHS
    model.fit(
        X,
        y,
        epochs=100,
        verbose=0
    )

    return model

# ---------------- INPUT METHOD ----------------
st.subheader("Choose Input Method:")

input_method = st.selectbox(
    "",
    ["Text Input"]
)

# ---------------- TEXT INPUT ----------------
seed_text = st.text_area(
    "Enter the Starting Text:",
    height=120
)

# ---------------- NUMBER OF WORDS ----------------
next_words = st.slider(
    "Choose the next words",
    1,
    20,
    10
)

# ---------------- GENERATE BUTTON ----------------
if st.button("🟩 Add Word"):

    with st.spinner("Training AI Model..."):
        model = train_model()

    generated_text = seed_text

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences(
            [generated_text]
        )[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding='pre'
        )

        predicted = np.argmax(
            model.predict(token_list, verbose=0),
            axis=-1
        )

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        # ---------------- REMOVE REPETITION ----------------
        words = generated_text.split()

        if output_word not in words[-3:]:
            generated_text += " " + output_word

    # ---------------- GENERATED TEXT ----------------
    st.subheader("📜 Final Generated Text:")

    st.text_area(
        "",
        generated_text,
        height=150
    )

    # ---------------- TRANSLATION ----------------
    translated = GoogleTranslator(
        source='auto',
        target=lang_codes[language]
    ).translate(generated_text)

    st.subheader(f"🟢 Translated Output ({language}):")

    st.write(translated)

    # ---------------- AUDIO ----------------
    tts = gTTS(
        translated,
        lang=lang_codes[language]
    )

    tts.save("output.mp3")

    audio_file = open("output.mp3", "rb")

    st.audio(audio_file.read())

    # ---------------- RESET ----------------
    if st.button("🗑 Reset"):
        st.session_state.clear()
        st.rerun()