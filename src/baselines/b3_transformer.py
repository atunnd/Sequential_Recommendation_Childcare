import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.config import (
    N_CAT, VOCAB_SIZE, CLS_ID, MASK_ID, PAD_ID, TOTAL_VOCAB,
    TRANS_D_MODEL, TRANS_N_HEADS, TRANS_N_LAYERS, TRANS_FF_DIM, TRANS_DROPOUT,
    MAM_MASK_RATE, MAM_EPOCHS, MAM_BATCH_SIZE, MAM_LR, RANDOM_SEED,
)


def compute_max_len(sequences_train: dict, quantile: float = 0.95) -> int:
    """Data-driven MAX_LEN: 95th percentile of sequence lengths in train split."""
    lengths = [len(v["activities"]) for v in sequences_train.values()]
    return int(np.quantile(lengths, quantile)) + 1  # +1 for CLS token


def tokenize(activities: list[int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = [CLS_ID] + activities
    tokens = tokens[:max_len]
    padding = max_len - len(tokens)
    tokens = tokens + [PAD_ID] * padding
    attn_mask = [1 if t != PAD_ID else 0 for t in tokens]
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long)


def mask_for_mam(
    tokens: torch.Tensor, mask_rate: float = MAM_MASK_RATE
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = tokens.clone()
    rand = torch.rand(tokens.shape)
    maskable = (tokens != CLS_ID) & (tokens != PAD_ID)
    to_mask = (rand < mask_rate) & maskable
    tokens = tokens.clone()
    tokens[to_mask] = MASK_ID
    labels[~to_mask] = -100     # CrossEntropyLoss ignores index -100
    return tokens, labels


class ATUSDataset(Dataset):
    def __init__(self, sequences: dict, max_len: int, augment: bool = True):
        self.ids = list(sequences.keys())
        self.sequences = sequences
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        key = self.ids[idx]
        acts = self.sequences[key]["activities"]
        tokens, attn_mask = tokenize(acts, self.max_len)
        if self.augment:
            tokens_masked, labels = mask_for_mam(tokens)
        else:
            tokens_masked, labels = tokens, torch.full_like(tokens, -100)
        return tokens_masked, attn_mask, labels, tokens


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = TOTAL_VOCAB,
        d_model: int = TRANS_D_MODEL,
        n_heads: int = TRANS_N_HEADS,
        n_layers: int = TRANS_N_LAYERS,
        ff_dim: int = TRANS_FF_DIM,
        dropout: float = TRANS_DROPOUT,
        max_len: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(tokens) + self.pos_embedding(positions)
        # TransformerEncoder expects src_key_padding_mask: True = ignore
        pad_mask = attn_mask == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x


class B3Model(nn.Module):
    def __init__(self, max_len: int = 64):
        super().__init__()
        self.encoder = TransformerEncoder(max_len=max_len)
        self.pred_head = nn.Linear(TRANS_D_MODEL, VOCAB_SIZE)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor):
        x = self.encoder(tokens, attn_mask)
        cls = x[:, 0, :]                        # CLS token representation
        logits = self.pred_head(x)              # (B, L, vocab)
        return cls, logits

    def embed(self, tokens: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.encoder(tokens, attn_mask)
            cls = x[:, 0, :]
            return F.normalize(cls, p=2, dim=-1)   # L2 normalize (fix)


class B3Pipeline:
    """Full B3 training + embedding pipeline."""

    def __init__(self):
        self.model: B3Model | None = None
        self.max_len: int | None = None

    def fit(self, sequences_train: dict, device: str = "cpu") -> "B3Pipeline":
        # Seed before building the model: covers weight init, MAM masking, and batch order.
        # Without this B3 is not reproducible across runs.
        torch.manual_seed(RANDOM_SEED)
        self.max_len = compute_max_len(sequences_train)
        self.model = B3Model(max_len=self.max_len).to(device)

        dataset = ATUSDataset(sequences_train, self.max_len, augment=True)
        loader = DataLoader(
            dataset, batch_size=MAM_BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=False
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=MAM_LR)

        self.model.train()
        for epoch in range(MAM_EPOCHS):
            total_loss = 0.0
            for tokens_masked, attn_mask, labels, _ in tqdm(
                loader, desc=f"B3 epoch {epoch+1}/{MAM_EPOCHS}", leave=False
            ):
                tokens_masked = tokens_masked.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)

                _, logits = self.model(tokens_masked, attn_mask)
                loss = F.cross_entropy(
                    logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  B3 epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

        return self

    def transform(self, sequences: dict, device: str = "cpu") -> tuple[list[str], np.ndarray]:
        assert self.model is not None, "Call fit() first"
        self.model.eval()
        ids = list(sequences.keys())
        dataset = ATUSDataset(sequences, self.max_len, augment=False)
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
        embeddings = []
        with torch.no_grad():
            for tokens_masked, attn_mask, _, tokens in loader:
                emb = self.model.embed(tokens.to(device), attn_mask.to(device))
                embeddings.append(emb.cpu().numpy())
        return ids, np.vstack(embeddings)

    def fit_transform(self, sequences_train: dict, device: str = "cpu"):
        self.fit(sequences_train, device)
        return self.transform(sequences_train, device)

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(), "max_len": self.max_len}, path)

    def load(self, path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device)
        self.max_len = ckpt["max_len"]
        self.model = B3Model(max_len=self.max_len).to(device)
        self.model.load_state_dict(ckpt["model"])
        return self
