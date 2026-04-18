# 🧠 AI Text Generator (LSTM)

An interactive Natural Language Processing (NLP) project that generates text using a trained Long Short-Term Memory (LSTM) neural network. The application predicts the next word in a sequence and can generate multiple words based on user input.

---

## 🚀 Features

* 🔤 Next-word prediction using LSTM
* ✨ Multi-word text generation
* 🎛 Adjustable creativity using temperature sampling
* 🖥 Interactive UI built with Streamlit
* ⚡ Fast inference with optimized word lookup

---

## 🧠 How It Works

1. Input text is tokenized using a pre-trained tokenizer
2. The sequence is padded to match the model’s input length
3. The LSTM model predicts the probability distribution of the next word
4. Temperature-based sampling selects the next word
5. The process repeats to generate multiple words

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Streamlit

---

## 📁 Project Structure

```
Next_Word_Pred/
│
├── app.py              # Streamlit application
├── lstm_model.h5       # Trained LSTM model
├── tokenizer.pkl       # Tokenizer used during training
├── max_len.pkl         # Maximum sequence length
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## ▶️ How to Run Locally

1. Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
```

2. Navigate to the project folder:

```
cd your-repo-name
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the application:

```
streamlit run app.py
```

---

## 📌 Notes

* The model is trained on a text dataset using sequence prediction techniques
* Output quality depends on training data size and preprocessing
* LSTM models have limitations in long-range context understanding compared to newer architectures

---

## 📚 Future Improvements

* Upgrade to Transformer-based architecture
* Improve dataset quality and size
* Add advanced decoding methods (Top-k / Top-p sampling)
* Enhance UI with additional visualizations

---

## 👨‍💻 Author

Developed as a mini NLP project to explore sequence modeling and text generation using deep learning.
