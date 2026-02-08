import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


print("\n========== Loading Dataset ==========")
from datasets import load_dataset

dataset = load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][:5])

texts = dataset["train"]["sentence"]
labels = dataset["train"]["label"]


# 85% train+val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    texts,
    labels,
    test_size=0.15,
    stratify=labels,
    random_state=42
)

# 85% train, 15% val (of the 85%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.15,
    stratify=y_temp,
    random_state=42
)

print("Loading FastText embeddings...")
ft = KeyedVectors.load_word2vec_format(
    "cc.en.300.vec",
    binary=False
)
EMBED_DIM = 300

def sentence_to_embedding(sentence, model):
    tokens = sentence.lower().split()
    vectors = []

    for tok in tokens:
        if tok in model:
            vectors.append(model[tok])

    if len(vectors) == 0:
        return np.zeros(EMBED_DIM)

    return np.mean(vectors, axis=0)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, ft_model):
        self.embeddings = [
            sentence_to_embedding(t, ft_model) for t in texts
        ]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

class_counts = torch.tensor([604, 2879, 1363], dtype=torch.float)
class_weights = class_counts.sum() / (3 * class_counts)

criterion = nn.CrossEntropyLoss(weight=class_weights)

train_ds = SentimentDataset(X_train, y_train, ft)
val_ds   = SentimentDataset(X_val, y_val, ft)
test_ds  = SentimentDataset(X_test, y_test, ft)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        return self.net(x)


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

    return total_loss / len(loader), acc, f1


model = MLPClassifier()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

best_val_f1 = 0.0

for epoch in range(20):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    val_loss, val_acc, val_f1 = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: Val F1 = {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_mlp.pth")


model.load_state_dict(torch.load("best_mlp.pth"))
test_loss, test_acc, test_f1 = evaluate(model, test_loader)

print("==== Test Results ====")
print(f"Accuracy: {test_acc:.4f}")
print(f"Macro F1: {test_f1:.4f}")
