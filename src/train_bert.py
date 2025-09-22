# train_bert_colab.py

import torch
from sklearn.metrics import f1_score
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# --- 1. Load Data ---
print("Loading data from Colab filesystem...")
train_dataset = load_from_disk("data/train/train")
val_dataset   = load_from_disk("data/val/val")
test_dataset  = load_from_disk("data/test/test")

# --- 2. Load Tokenizer and Model ---
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# --- 3. Tokenize Datasets ---
def tokenize_function(examples):
    return tokenizer(
        examples["content"],
        padding="max_length",  # pad all sequences to max_length
        truncation=True,
        max_length=256        # or another sensible max length
    )

print("Tokenizing datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Rename label column to 'labels'
tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
tokenized_val_dataset = tokenized_val_dataset.rename_column("label", "labels")

# Remove unnecessary columns safely
cols_to_remove = [c for c in ["content", "title", "reviewText", "summary"] 
                  if c in tokenized_train_dataset.column_names]
tokenized_train_dataset = tokenized_train_dataset.remove_columns(cols_to_remove)
tokenized_val_dataset = tokenized_val_dataset.remove_columns(cols_to_remove)

# --- 4. Define Training Arguments ---
training_args = TrainingArguments(
    output_dir="./models/bert_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=50,         # log every 50 steps
    report_to="none",         # disable extra logging
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=False,          # Minimalistic, standard 32-bit
    optim="adamw_torch", # Stable optimizer for GPU
)

# --- 5. Metrics ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"f1": f1_score(labels, predictions)}

# --- 6. Initialize Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
)

# --- 7. Start Training ---
print("Starting BERT training...")
trainer.train()

# --- 8. Save Model and Tokenizer ---
print("Training complete. Saving model and tokenizer...")
trainer.save_model("/content/drive/MyDrive/bert_model")
tokenizer.save_pretrained("/content/drive/MyDrive/bert_model")
print("Model saved. You can now zip and download it from Colab.")
