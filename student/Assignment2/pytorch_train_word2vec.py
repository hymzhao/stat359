import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive


# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, df):
        """
        df: DataFrame with columns ['center', 'context']
        """
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        center = row['center']
        context = row['context']
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)

# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.in_embeddings.weight)
        nn.init.xavier_uniform_(self.out_embeddings.weight)

    def forward(self, center, context):
        """
        center: [batch_size]
        context: [batch_size]
        Returns: dot product of embeddings [batch_size]
        """
        center_embeds = self.in_embeddings(center)  # [batch_size, embedding_dim]
        context_embeds = self.out_embeddings(context)  # [batch_size, embedding_dim]
        # Dot product for each pair
        dot = torch.sum(center_embeds * context_embeds, dim=1)
        return dot

    def get_embeddings(self):
        # Return input embeddings as NumPy
        return self.in_embeddings.weight.data.cpu().numpy()


# Load processed data
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

skipgram_df = data['skipgram_df']
vocab_size = len(data['word2idx'])
counter = data['counter']

# Precompute negative sampling distribution below
# Step 1: Negative sampling distribution
# Get counts aligned to idx2word
word_counts = torch.tensor([counter[data['idx2word'][i]] for i in range(vocab_size)], dtype=torch.float)

# Apply 3/4 power smoothing
word_probs = word_counts.pow(0.75)

# Normalize to sum to 1
word_probs /= word_probs.sum()


# Device selection: CUDA > MPS > CPU
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else
                      'cpu')
print("Using device:", device)

# Dataset and DataLoader
# Create dataset
dataset = SkipGramDataset(skipgram_df)

# Create DataLoader (shuffles the data for each epoch)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Model, Loss, Optimizer
# Initialize model
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)

# Loss function: BCEWithLogitsLoss for positive & negative samples
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def make_targets(center, context, vocab_size):
    """
    Create label tensors for positive and negative samples.
    
    Args:
        center: tensor of shape [batch_size] with center word indices
        context: tensor of shape [batch_size] with positive context word indices
        vocab_size: total vocabulary size (not used in label creation here)
    
    Returns:
        pos_labels: tensor of 1s for positive pairs [batch_size]
        neg_labels: tensor of 0s for negative pairs [batch_size * NEGATIVE_SAMPLES]
    """
    batch_size = center.size(0)
    pos_labels = torch.ones(batch_size, device=center.device)
    neg_labels = torch.zeros(batch_size * NEGATIVE_SAMPLES, device=center.device)
    return pos_labels, neg_labels


# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for centers, contexts in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        centers = centers.to(device)
        contexts = contexts.to(device)

        # Positive samples
        pos_scores = model(centers, contexts)
        pos_labels, neg_labels = make_targets(centers, contexts, vocab_size)

        # Negative sampling
        neg_contexts = torch.multinomial(word_probs, len(centers) * NEGATIVE_SAMPLES, replacement=True).to(device)
        neg_centers = centers.repeat_interleave(NEGATIVE_SAMPLES)
        neg_scores = model(neg_centers, neg_contexts)

        # Combine positive and negative samples
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])

        # Compute loss and backpropagate
        loss = criterion(all_scores, all_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings,
                 'word2idx': data['word2idx'],
                 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
