# evaluate.py

import torch
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import json

# --- 1. Load Test Data ---
print("Loading test dataset...")
test_dataset = load_from_disk("data/test/test")  # Adjust path if needed

# --- 2. Load DistilBERT Model & Tokenizer ---
print("Loading DistilBERT model...")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/bert_model")
model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/bert_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- 3. Tokenize Test Dataset ---
def tokenize_function(examples):
    return tokenizer(examples["content"], padding=True, truncation=True)

print("Tokenizing test dataset...")
tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test = tokenized_test.rename_column("label", "labels")

# Remove unnecessary columns safely
cols_to_remove = [c for c in ["content", "title", "reviewText", "summary"] if c in tokenized_test.column_names]
tokenized_test = tokenized_test.remove_columns(cols_to_remove)
# --- 4. Inference ---
batch_size = 16
dataloader = DataLoader(tokenized_test, batch_size=batch_size)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        # Move tensors to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # --- 5. Metrics ---
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")
cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# --- 6. Save Metrics ---
metrics = {
    "accuracy": acc,
    "f1_score": f1,
    "confusion_matrix": cm.tolist()  # convert numpy array to list
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to models/metrics.json")
