# TextOracle: Advanced Predictive Text Generation for Dynamic Communication

## Project Overview

TextOracle is an AI-powered predictive text generation application developed using Python, TensorFlow/Keras, NLP techniques, and Streamlit.

The application uses an **LSTM-based neural network** to generate additional text based on user-provided starting text. The generated text can also be translated into multiple languages and converted into speech.

## Objective

The main objectives of TextOracle are to:

- Generate predictive text from user-provided starting text
- Apply Natural Language Processing techniques for text preparation
- Use an LSTM neural network for next-word prediction
- Provide a simple interactive Streamlit interface
- Support multilingual translated output
- Convert generated translated text into speech

## Technologies Used

- Python
- TensorFlow
- Keras
- NLP
- Pandas
- NumPy
- Streamlit
- Deep Translator
- Google Text-to-Speech (gTTS)

## Model Architecture

The predictive text generation model uses the following workflow:

```text
Input Text
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
N-Gram Sequence Generation
    ↓
Sequence Padding
    ↓
Embedding Layer
    ↓
LSTM Layer
    ↓
Dense + Softmax
    ↓
Next-Word Prediction
    ↓
Generated Text
```

### Neural Network Components

- **Embedding Layer:** 100-dimensional word embeddings
- **LSTM Layer:** 200 units
- **Dense Layer:** Softmax activation for word prediction
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Training:** 100 epochs

## Dataset

The application uses a dataset named:

```text
Text_Generation_831_rows.csv
```

The application reads the `headline` column from the dataset and uses the text for model training and predictive text generation.

The dataset contains **831 rows**.

## Text Processing

The application performs several text preprocessing steps:

- Removes punctuation
- Converts text to lowercase
- Removes unsupported characters
- Tokenizes the text
- Creates n-gram sequences
- Pads sequences to a common length

The prepared sequences are then divided into input sequences and target words for model training.

## Predictive Text Generation

Users can enter starting text through the Streamlit interface and select the number of words to generate.

The application supports generating between:

**1 and 20 next words**

The model predicts the next word iteratively and appends the predicted words to the starting text.

A repetition check is also applied to reduce repeated words in the generated output.

## Multilingual Output

TextOracle supports translated output in:

- English
- Hindi
- Telugu
- Tamil

The generated text is translated using the **Deep Translator** library.

## Text-to-Speech

After translation, the application uses **Google Text-to-Speech (gTTS)** to convert the translated text into audio.

The generated audio can be played directly within the Streamlit application.

## Streamlit Application

The application provides:

- Language selection
- Starting text input
- Number of words selection
- Predictive text generation
- Generated text display
- Multilingual translation
- Audio output

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Manjusha266/TextOracle-Predictive-Text-Generation.git
```

### 2. Navigate to the Project Folder

```bash
cd TextOracle-Predictive-Text-Generation
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Make Sure the Dataset Is Available

Place the following file in the project directory:

```text
Text_Generation_831_rows.csv
```

The application expects the dataset to be available in the same directory as `app.py`.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in the browser.

## Project Structure

```text
TextOracle-Predictive-Text-Generation/
│
├── app.py
├── requirements.txt
├── Text_Generation_831_rows.csv
└── README.md
```

## Skills Demonstrated

This project demonstrates practical skills in:

- Python Programming
- Natural Language Processing
- TensorFlow/Keras
- LSTM Neural Networks
- Text Preprocessing
- Tokenization
- N-Gram Sequence Generation
- Predictive Text Generation
- Streamlit Application Development
- Multilingual Translation
- Text-to-Speech
- Data Handling with Pandas and NumPy

## Key Features

- AI-powered predictive text generation
- LSTM-based next-word prediction
- Interactive Streamlit interface
- Adjustable number of generated words
- English, Hindi, Telugu, and Tamil output
- Text translation
- Text-to-speech conversion
- Repetition reduction during generation

## Conclusion

TextOracle demonstrates an end-to-end NLP application that combines text preprocessing, sequence generation, LSTM-based deep learning, multilingual translation, and text-to-speech capabilities.

The project helped demonstrate practical experience in Python, TensorFlow/Keras, Natural Language Processing, neural networks, and Streamlit application development.
