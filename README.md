# Large-scale-sentiment-analysis-bert

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/Hugging_Face-FP16_BERT-orange)](https://huggingface.co/)

This repository contains a sentiment analysis system for Amazon product reviews using two models: a **lightweight BERT model (FP16)** and a **custom LSTM model**. It provides a Gradio-based interface for easy predictions.

---

## Features

- **BERT (FP16)**: Fine-tuned BERT model converted to half-precision for smaller size and faster inference.
- **LSTM**: Bidirectional LSTM network trained on the same dataset for comparison.
- **Gradio Interface**: Web UI to input a review and select the model for sentiment prediction.
- **Lightweight & Efficient**: Only the FP16 BERT model is included; original full BERT files are excluded.

---

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/tanishqkz/Amazon-Review-Sentiment-Classifier.git
    cd Amazon-Review-Sentiment-Classifier
    ```

2.  **Create a Python environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1.  **Launch the Gradio app:**

    ```bash
    python src/app.py
    ```

2.  Enter an Amazon review.

3.  Select a model: **BERT** or **LSTM**.

4.  Click **Predict** to see the sentiment.

---

## File Structure

```
src/
├── app.py                  # Gradio application
├── shrink.py               # Script to convert BERT to FP16
├── train_lstm.py           # Training script for LSTM
├── models/
│   ├── bert_model/
│   │   └── model_fp16.safetensors  # Lightweight BERT
│   └── lstm_model.pt
└── data/                   # Train/test/validation datasets
```

---

## Notes

- The FP16 BERT model is tracked via **Git LFS**.
- Original full-precision BERT files are excluded to keep the repository size manageable.
- Ensure your system has sufficient RAM/VRAM for inference.

---

## License

MIT License. See `LICENSE` for details.
