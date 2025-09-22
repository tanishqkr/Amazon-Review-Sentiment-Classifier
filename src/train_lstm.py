import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import os
import time

# --- 1. Create a custom Dataset for tokenization ---
class TextDataset(Dataset):
    def __init__(self, dataset, vocab):
        self.texts = dataset['content'] # Corrected column name
        self.labels = dataset['label']
        self.vocab = vocab
        self.pad_idx = vocab['<PAD>']
        self.unk_idx = vocab['<UNK>']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Simple tokenization and numericalization
        tokens = text.lower().split()
        numericalized_tokens = [self.vocab.get(token, self.unk_idx) for token in tokens]
        
        return torch.tensor(numericalized_tokens), torch.tensor(label)

# --- 2. Define the LSTM Model ---
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
        
        
        # Output dim is 1 for binary classification
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        
        output, (hidden, cell) = self.lstm(embedded)

        # Concatenate the final forward and backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) if self.lstm.bidirectional else hidden[-1,:,:])

        return self.fc(hidden) # This output has shape [batch_size, 1]

# --- 3. Main Training Function ---
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_preds = 0
        total_preds = 0

        for batch in train_loader:
            text, labels = batch
            text = text.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            predictions = model(text).squeeze(1) # <<-- FIX: squeeze the output here
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # Calculate accuracy
            predicted_labels = (torch.sigmoid(predictions) > 0.5).int()
            correct_preds += (predicted_labels == labels.int()).sum().item()
            total_preds += labels.size(0)

        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                text, labels = batch
                text = text.to(device)
                labels = labels.to(device).float()
                
                predictions = model(text).squeeze(1) # <<-- FIX: squeeze the output here
                loss = criterion(predictions, labels)
                val_loss += loss.item()

                predicted_labels = (torch.sigmoid(predictions) > 0.5).int()
                val_correct += (predicted_labels == labels.int()).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        print(f"Epoch: {epoch+1}/{num_epochs} | Training Loss: {epoch_loss/len(train_loader):.4f} | Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

# --- 4. Main script execution ---
if __name__ == '__main__':
    # Set device to GPU if available (for M1 Macs, this is "mps")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data from disk (prepared by preprocess.py)
    train_data = load_from_disk("data/train")
    val_data = load_from_disk("data/val")
    
    # Build vocabulary from training data
    print("Building vocabulary...")
    all_tokens = [token for text in train_data['content'] for token in text.lower().split()]
    vocab_counter = Counter(all_tokens)
    vocab = {word: i + 2 for i, (word, _) in enumerate(vocab_counter.most_common())}
    vocab['<UNK>'] = 1
    vocab['<PAD>'] = 0
    print(f"Vocabulary built with {len(vocab)} unique tokens.")
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_data, vocab)
    val_dataset = TextDataset(val_data, vocab)

    def collate_fn(batch):
        # Collate function to handle variable-length sequences
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=vocab['<PAD>'])
        return padded_texts, torch.stack(labels)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Hyperparameters
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-3
    
    # Instantiate model, loss, and optimizer
    model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting LSTM training...")
    train_model(model, train_loader, val_loader, optimizer, criterion, device, NUM_EPOCHS)
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_model.pt")
    print("Training complete. Model saved to 'models/lstm_model.pt'.")