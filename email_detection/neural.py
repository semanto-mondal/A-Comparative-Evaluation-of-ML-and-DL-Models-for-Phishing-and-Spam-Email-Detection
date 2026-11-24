"""BiLSTM and Transformer encoder classifiers (PyTorch), trained from scratch."""

import gc
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import set_seed

PAD, UNK = 0, 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Vocabulary and encoding
# ---------------------------------------------------------------------------

def build_vocab(texts, max_size: int) -> dict:
    """Most frequent training words (seen at least twice); ids 0 and 1 are PAD and UNK."""
    counts = Counter(w for t in texts for w in t.split())
    words = [w for w, c in counts.most_common(max_size - 2) if c >= 2]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(texts, vocab: dict, maxlen: int) -> np.ndarray:
    """Word ids, truncated to the first maxlen words and padded with zeros."""
    ids = np.zeros((len(texts), maxlen), dtype=np.int64)
    for i, text in enumerate(texts):
        seq = [vocab.get(w, UNK) for w in text.split()[:maxlen]]
        ids[i, :len(seq)] = seq
    return ids


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=64, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        mask = x != PAD
        mask[:, 0] = True                     # empty emails: pool over the first step
        h, _ = self.lstm(self.drop(self.emb(x)))
        h = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values   # masked max-pooling
        return self.head(h).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, maxlen=250, d_model=128, heads=4, ff=256, layers=2, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(maxlen, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        pad = x == PAD
        pad[:, 0] = False                     # avoid fully masked rows
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.encoder(self.tok(x) + self.pos(positions), src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        h = (h * keep).sum(1) / keep.sum(1).clamp(min=1)   # masked mean-pooling
        return self.head(h).squeeze(-1)


def build_sequence_model(name, vocab_size, cfg):
    if name == "BiLSTM":
        return BiLSTMClassifier(vocab_size, emb_dim=cfg.seq_emb)
    if name == "Transformer":
        return TransformerClassifier(vocab_size, maxlen=cfg.seq_maxlen, d_model=cfg.seq_emb)
    raise ValueError(f"unknown sequence model: {name}")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_logits(model, loader) -> np.ndarray:
    model.eval()
    return torch.cat([model(xb.to(DEVICE)).float().cpu() for xb, _ in loader]).numpy()


def _loader(df, text_col, vocab, cfg, shuffle, seed):
    x = torch.from_numpy(encode(df[text_col].tolist(), vocab, cfg.seq_maxlen))
    y = torch.tensor(df.label.values, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=cfg.seq_batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def run_sequence_model(store, cfg, exp, name, seed, train, val, test, text_col="x_seq", keep_model=False):
    """Adam + binary cross-entropy; early stopping on validation loss, best epoch restored."""
    if store.done(exp, name, seed):
        return
    set_seed(seed)

    start = time.time()
    vocab = build_vocab(train[text_col], cfg.seq_vocab)          # training split only
    train_dl = _loader(train, text_col, vocab, cfg, True, seed)
    val_dl = _loader(val, text_col, vocab, cfg, False, seed)
    test_dl = _loader(test, text_col, vocab, cfg, False, seed)

    model = build_sequence_model(name, len(vocab) + 2, cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.seq_lr)
    loss_fn = nn.BCEWithLogitsLoss()
    y_val = torch.tensor(val.label.values, dtype=torch.float32)

    history = defaultdict(list)
    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(cfg.seq_max_epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += ((logits > 0).float() == yb).sum().item()
            seen += len(yb)

        val_logits = predict_logits(model, val_dl)
        val_loss = loss_fn(torch.tensor(val_logits), y_val).item()
        history["train_loss"].append(total_loss / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(((val_logits > 0).astype(int) == val.label.values).mean()))

        if val_loss < best_loss - 1e-4:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch + 1
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.seq_patience:
                break

    model.load_state_dict(best_state)
    train_time = time.time() - start

    start = time.time()
    score = 1 / (1 + np.exp(-predict_logits(model, test_dl)))
    infer_time = time.time() - start

    store.save(exp, name, seed, test, score, (score >= 0.5).astype(int), train_time, infer_time,
               n_params=count_parameters(model), history=dict(history))
    if keep_model:
        torch.save({"state": best_state, "vocab": vocab}, store.model_path(exp, name, seed, ".pt"))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
