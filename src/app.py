# app.py
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from collections import Counter

# --- 1. Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Load Shrunk BERT Model ---
bert_tokenizer = AutoTokenizer.from_pretrained("models/bert_model")
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "models/bert_model", torch_dtype=torch.float16
)
bert_model.to(device)
bert_model.eval()

# --- 3. LSTM Model Definition ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) if self.lstm.bidirectional else hidden[-1,:,:])
        return self.fc(hidden)

# --- 4. Rebuild vocab exactly as in training ---
train_data = load_from_disk("data/train")
all_tokens = [token for text in train_data['content'] for token in text.lower().split()]
vocab_counter = Counter(all_tokens)
vocab = {word: i + 2 for i, (word, _) in enumerate(vocab_counter.most_common())}
vocab['<UNK>'] = 1
vocab['<PAD>'] = 0
VOCAB_SIZE = len(vocab)

# --- 5. Instantiate and load LSTM model ---
lstm_model = LSTMClassifier(
    vocab_size=VOCAB_SIZE,
    embedding_dim=100,
    hidden_dim=256,
    output_dim=1,
    n_layers=2,
    bidirectional=True,
    dropout=0.5
).to(device)

lstm_model.load_state_dict(torch.load("models/lstm_model.pt", map_location=device))
lstm_model.eval()

# --- 6. LSTM preprocessing ---
def tokenize_lstm(text):
    tokens = text.lower().split()
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)  # batch dimension

# --- 7. Prediction function ---
def predict_sentiment(text, model_choice):
    if not model_choice:
        return "Please select a model first!"
    
    if model_choice == "BERT":
        inputs = bert_tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        positive_prob = probs[0, 1].item()
        negative_prob = probs[0, 0].item()
        pred_class = "Positive" if positive_prob > negative_prob else "Negative"
        confidence = max(positive_prob, negative_prob) * 100
        return f"{pred_class} ({confidence:.2f}%)"
    
    else:  # LSTM
        indices = tokenize_lstm(text)
        with torch.no_grad():
            output = lstm_model(indices).squeeze(1)
            prob = torch.sigmoid(output).item()
            pred_label = 1 if prob > 0.5 else 0
        pred_class = "Positive" if pred_label == 1 else "Negative"
        return f"{pred_class} ({prob*100:.2f}%)"

# --- 8. Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style="
            background: linear-gradient(135deg, #6D5BBA, #8D58BF);
            padding: 2rem;
            border-radius: 12px;
            text-align:center;
            color: white;
        ">
        <h1>Sentiment Analyzer</h1>
        <p>Enter a review and choose a model to predict sentiment!</p>
        </div>
        """,
        elem_id="header"
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(label="Your Review", placeholder="Type here...")
            model_choice = gr.Dropdown(choices=["BERT", "LSTM"], label="Select Model", value=None)
            submit_btn = gr.Button("Predict")

        with gr.Column(scale=1):
            output = gr.Label(label="Sentiment")

    submit_btn.click(fn=predict_sentiment, inputs=[text_input, model_choice], outputs=output)

# --- 9. Launch ---
demo.launch()
