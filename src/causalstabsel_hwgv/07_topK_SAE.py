# Step 7: Train a Top-K sparse autoencoder on the pooled satellite
# embeddings from 06 (every year combined) to learn a shared, interpretable
# feature dictionary, then encode every ZCTA-year through it and save the
# resulting sparse atom activations to data/satellite_sae_features_2014_2022.csv.

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.preprocessing import StandardScaler
import joblib


# ==========================
# Configuration
# ==========================

# The SAE dictionary is trained once on every available year (2014-2022),
# not per year: 06_sat_embedder.py already row-binds every year into one
# combined embeddings file, so this just reads that single file.
INPUT_FILE = "data/satellite_embeddings_2014_2022.csv"

OUTPUT_FEATURES = "data/satellite_sae_features_2014_2022.csv"

OUTPUT_MODEL = "data/models/satellite_sae_model.pt"

OUTPUT_SCALER = "data/models/satellite_embedding_scaler.pkl"


INPUT_DIM = 768

# Number of SAE dictionary atoms
SAE_DIM = 4096

# Number of active atoms per image
TOP_K = 25


EPOCHS = 500
BATCH_SIZE = 32
LR = 1e-3


TEST_SIZE = 0.2


# ==========================
# Create folders
# ==========================

os.makedirs(
    "data/models",
    exist_ok=True
)


# ==========================
# Load embeddings
# ==========================

df = pd.read_csv(INPUT_FILE)


print("Loaded:")
print(df.shape)


metadata = df[
    [
        "city",
        "year",
        "zcta",
        "filename"
    ]
]


embedding_cols = [
    f"emb_{i}"
    for i in range(INPUT_DIM)
]


X = df[embedding_cols].values



# ==========================
# Normalize embeddings
# ==========================

scaler = StandardScaler()

X = scaler.fit_transform(X)


joblib.dump(
    scaler,
    OUTPUT_SCALER
)


print("Saved scaler")


X = torch.tensor(
    X,
    dtype=torch.float32
)



# ==========================
# Dataset split
# ==========================

dataset = TensorDataset(X)


train_size = int(
    len(dataset) * (1 - TEST_SIZE)
)

test_size = len(dataset) - train_size


train_dataset, test_dataset = random_split(
    dataset,
    [
        train_size,
        test_size
    ]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)



print(
    f"Train: {train_size}, Test: {test_size}"
)



# ==========================
# TopK Sparse Autoencoder
# ==========================

class TopKSAE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        k
    ):
        super().__init__()

        self.encoder = nn.Linear(
            input_dim,
            hidden_dim
        )


        self.decoder = nn.Linear(
            hidden_dim,
            input_dim,
            bias=False
        )


        self.k = k



    def top_k(self, x):

        values, indices = torch.topk(
            x,
            self.k,
            dim=1
        )


        sparse = torch.zeros_like(x)


        sparse.scatter_(
            1,
            indices,
            values
        )


        return sparse



    def encode(self, x):

        z = self.encoder(x)

        z = torch.relu(z)

        z = self.top_k(z)

        return z



    def forward(self, x):

        z = self.encode(x)

        reconstruction = self.decoder(z)

        return reconstruction, z



# ==========================
# Initialize SAE
# ==========================

model = TopKSAE(
    INPUT_DIM,
    SAE_DIM,
    TOP_K
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)



# ==========================
# Train
# ==========================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0


    for (batch,) in train_loader:


        optimizer.zero_grad()


        reconstruction, z = model(batch)


        loss = torch.mean(
            (reconstruction - batch) ** 2
        )


        loss.backward()

        optimizer.step()


        train_loss += loss.item()



    # validation

    model.eval()

    test_loss = 0


    with torch.no_grad():

        for (batch,) in test_loader:

            reconstruction, z = model(batch)

            loss = torch.mean(
                (reconstruction - batch) ** 2
            )

            test_loss += loss.item()



    if epoch % 50 == 0:

        print(
            f"""
Epoch {epoch}
Train loss:
{train_loss / len(train_loader):.6f}

Test loss:
{test_loss / len(test_loader):.6f}
"""
        )



# ==========================
# Save model
# ==========================

torch.save(
    model.state_dict(),
    OUTPUT_MODEL
)


print(
    "Saved SAE:",
    OUTPUT_MODEL
)



# ==========================
# Generate SAE features
# ==========================

model.eval()


with torch.no_grad():

    sparse_features = model.encode(X)


sparse_features = (
    sparse_features
    .numpy()
)



print(
    "SAE feature shape:",
    sparse_features.shape
)



# ==========================
# Save atom activations
# ==========================

atom_cols = [
    f"atom_{i}"
    for i in range(SAE_DIM)
]


sae_df = pd.DataFrame(
    sparse_features,
    columns=atom_cols
)


final_df = pd.concat(
    [
        metadata.reset_index(drop=True),
        sae_df
    ],
    axis=1
)



final_df.to_csv(
    OUTPUT_FEATURES,
    index=False
)


print(
    "Saved:",
    OUTPUT_FEATURES
)