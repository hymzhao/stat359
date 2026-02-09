import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Reproducibility
# =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =====================
# Load dataset
# =====================
print("\n========== Loading Dataset ==========")
dataset = load_dataset("financial_phrasebank", "sentences_50agree")
print("Dataset loaded. Example:", dataset["train"][:5])

texts = dataset["train"]["sentence"]
labels = dataset["train"]["label"]

# =====================
# Train / Val / Test split (stratified)
# =====================
X_temp, X_test, y_temp, y_test = train_test_split(
    texts,
    labels,
    test_size=0.15,
    stratify=labels,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.15,
    stratify=y_temp,
    random_state=42
)

# =====================
# Load FastText
# =====================
print("Loading FastText embeddings...")
ft = KeyedVectors.load_word2vec_format("cc.en.300.vec", binary=False)

EMBED_DIM = 300
MAX_LEN = 32

# =====================
# Sentence → padded sequence
# =====================
def sentence_to_sequence(sentence, model):
    tokens = sentence.lower().split()
    vectors = []

    for tok in tokens[:MAX_LEN]:
        if tok in model:
            vectors.append(model[tok])
        else:
            vectors.append(np.zeros(EMBED_DIM))

    while len(vectors) < MAX_LEN:
        vectors.append(np.zeros(EMBED_DIM))

    return np.stack(vectors)  # (32, 300)

# =====================
# Dataset
# =====================
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, ft_model):
        self.sequences = [sentence_to_sequence(t, ft_model) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

train_ds = SentimentDataset(X_train, y_train, ft)
val_ds   = SentimentDataset(X_val, y_val, ft)
test_ds  = SentimentDataset(X_test, y_test, ft)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

# =====================
# Class weights
# =====================
class_counts = torch.tensor([604, 2879, 1363], dtype=torch.float)
class_weights = class_counts.sum() / (3 * class_counts)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# =====================
# LSTM Model
# =====================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128*2, 3)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h_final = self.dropout(h_final)
        return self.fc(h_final)

# =====================
# Evaluation
# =====================
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), acc, f1, all_labels, all_preds

# =====================
# Training
# =====================
model = LSTMClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_val_f1 = 0.0
EPOCHS = 30

# Store metrics for plotting
train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_f1s, val_f1s = [], []

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    # Evaluate both train and val
    train_loss, train_acc, train_f1, _, _ = evaluate(model, train_loader)
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_f1s.append(train_f1)
    val_f1s.append(val_f1)

    print(f"Epoch {epoch+1}: Train F1={train_f1:.4f} | Val F1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_lstm.pth")

# =====================
# Plot metrics
# =====================
plt.plot(range(1,EPOCHS+1), train_losses, label="Train Loss")
plt.plot(range(1,EPOCHS+1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.close()

plt.plot(range(1,EPOCHS+1), train_accs, label="Train Acc")
plt.plot(range(1,EPOCHS+1), val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_curve.png")
plt.close()

plt.plot(range(1,EPOCHS+1), train_f1s, label="Train F1")
plt.plot(range(1,EPOCHS+1), val_f1s, label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.legend()
plt.savefig("f1_curve.png")
plt.close()

# =====================
# Test
# =====================
model.load_state_dict(torch.load("best_lstm.pth"))
test_loss, test_acc, test_f1, test_labels, test_preds = evaluate(model, test_loader)

print("\n==== Test Results ====")
print(f"Accuracy: {test_acc:.4f}")
print(f"Macro F1: {test_f1:.4f}")

# =====================
# Confusion Matrix
# =====================
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg","Neu","Pos"], yticklabels=["Neg","Neu","Pos"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()
