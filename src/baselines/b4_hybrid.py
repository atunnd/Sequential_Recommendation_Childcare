import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    N_CAT, VOCAB_SIZE, TOTAL_VOCAB, CLS_ID, MASK_ID, PAD_ID,
    TRANS_D_MODEL, TRANS_N_HEADS, TRANS_N_LAYERS, TRANS_FF_DIM, TRANS_DROPOUT,
    MARKOV_HIDDEN, FUSED_DIM,
    MAM_MASK_RATE, MAM_EPOCHS, MAM_BATCH_SIZE, MAM_LR, RANDOM_SEED,
)
from src.baselines.b2_markov import compute_b2_raw
from src.baselines.b3_transformer import (
    TransformerEncoder, ATUSDataset, compute_max_len, mask_for_mam
)


class HybridDataset(ATUSDataset):
    """Dataset that also returns the raw unmasked activities for Markov branch."""

    def __getitem__(self, idx):
        key = self.ids[idx]
        seq = self.sequences[key]
        acts = seq["activities"]
        tokens, attn_mask = self._tokenize(acts)

        markov_feat = torch.tensor(
            compute_b2_raw(acts, N_CAT), dtype=torch.float32
        )

        if self.augment:
            tokens_masked, labels = mask_for_mam(tokens)
        else:
            tokens_masked, labels = tokens, torch.full_like(tokens, -100)

        return tokens_masked, attn_mask, labels, tokens, markov_feat

    def _tokenize(self, activities):
        from src.baselines.b3_transformer import tokenize
        return tokenize(activities, self.max_len)


class HybridEncoder(nn.Module):
    def __init__(self, max_len: int = 64):
        super().__init__()
        self.markov_mlp = nn.Sequential(
            nn.Linear(N_CAT * N_CAT, MARKOV_HIDDEN),
            nn.ReLU(),
        )
        self.transformer = TransformerEncoder(
            vocab_size=TOTAL_VOCAB,
            d_model=TRANS_D_MODEL,
            n_heads=TRANS_N_HEADS,
            n_layers=TRANS_N_LAYERS,
            ff_dim=TRANS_FF_DIM,
            dropout=TRANS_DROPOUT,
            max_len=max_len,
        )
        self.fusion = nn.Linear(MARKOV_HIDDEN + TRANS_D_MODEL, FUSED_DIM)
        self.pred_head = nn.Linear(TRANS_D_MODEL, VOCAB_SIZE)

    def forward(
        self,
        tokens_masked: torch.Tensor,
        attn_mask: torch.Tensor,
        markov_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        m = self.markov_mlp(markov_feat)                    # (B, MARKOV_HIDDEN)
        x = self.transformer(tokens_masked, attn_mask)      # (B, L, D_MODEL)
        t = x[:, 0, :]                                      # CLS embedding (B, D_MODEL)
        fused = self.fusion(torch.cat([m, t], dim=-1))
        e_u = F.normalize(fused, p=2, dim=-1)              # normalized fused embedding
        logits = self.pred_head(x)                          # (B, L, VOCAB_SIZE)
        return e_u, t, logits

    def embed(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        markov_feat: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            e_u, _, _ = self.forward(tokens, attn_mask, markov_feat)
            return e_u


class B4Pipeline:
    def __init__(self):
        self.model: HybridEncoder | None = None
        self.max_len: int | None = None

    def fit(self, sequences_train: dict, device: str = "cpu") -> "B4Pipeline":
        torch.manual_seed(RANDOM_SEED)
        self.max_len = compute_max_len(sequences_train)
        self.model = HybridEncoder(max_len=self.max_len).to(device)

        dataset = HybridDataset(sequences_train, self.max_len, augment=True)
        loader = DataLoader(
            dataset, batch_size=MAM_BATCH_SIZE, shuffle=True, num_workers=0
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=MAM_LR)

        self.model.train()
        for epoch in range(MAM_EPOCHS):
            total_loss = 0.0
            for tokens_masked, attn_mask, labels, _, markov_feat in tqdm(
                loader, desc=f"B4 epoch {epoch+1}/{MAM_EPOCHS}", leave=False
            ):
                tokens_masked = tokens_masked.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                markov_feat = markov_feat.to(device)

                _, _, logits = self.model(tokens_masked, attn_mask, markov_feat)
                loss = F.cross_entropy(
                    logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  B4 epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

        return self

    def transform(self, sequences: dict, device: str = "cpu") -> tuple[list[str], np.ndarray]:
        assert self.model is not None, "Call fit() first"
        self.model.eval()
        ids = list(sequences.keys())
        dataset = HybridDataset(sequences, self.max_len, augment=False)
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
        embeddings = []
        with torch.no_grad():
            for tokens_masked, attn_mask, _, tokens, markov_feat in loader:
                e_u = self.model.embed(
                    tokens.to(device), attn_mask.to(device), markov_feat.to(device)
                )
                embeddings.append(e_u.cpu().numpy())
        return ids, np.vstack(embeddings)

    def fit_transform(self, sequences_train: dict, device: str = "cpu"):
        self.fit(sequences_train, device)
        return self.transform(sequences_train, device)

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(), "max_len": self.max_len}, path)

    def load(self, path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device)
        self.max_len = ckpt["max_len"]
        self.model = HybridEncoder(max_len=self.max_len).to(device)
        self.model.load_state_dict(ckpt["model"])
        return self
