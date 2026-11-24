"""Fine-tuning BERT-base-uncased for legitimate vs malicious classification."""

import gc
import os
import time
from collections import defaultdict

# PyTorch only: stop transformers from importing TensorFlow, which can crash on
# environments where TensorFlow and NumPy versions do not match.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
from sklearn.metrics import f1_score                                  # noqa: E402
from torch.utils.data import DataLoader, Dataset                      # noqa: E402
from transformers import (AutoModelForSequenceClassification,         # noqa: E402
                          AutoTokenizer, get_linear_schedule_with_warmup)

from .evaluation import set_seed                                      # noqa: E402
from .neural import DEVICE, count_parameters                          # noqa: E402

SPECIAL_TOKENS = ["[URL]", "[EMAIL]"]
_tokenizers = {}


def get_tokenizer(name: str):
    """BERT tokenizer with [URL] and [EMAIL] added as single, never-split tokens."""
    if name not in _tokenizers:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        _tokenizers[name] = tokenizer
    return _tokenizers[name]


class _Encoded(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids, self.labels = input_ids, labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return self.input_ids[i], self.labels[i]


def make_loader(texts, labels, tokenizer, cfg, shuffle, seed):
    """Truncate to cfg.bert_maxlen tokens; pad per batch (faster than padding everything to the max)."""
    encoded = tokenizer(list(texts), truncation=True, max_length=cfg.bert_maxlen)

    def collate(batch):
        ids, y = zip(*batch)
        padded = tokenizer.pad({"input_ids": list(ids)}, return_tensors="pt")
        return padded["input_ids"], padded["attention_mask"], torch.tensor(y, dtype=torch.long)

    return DataLoader(_Encoded(encoded["input_ids"], labels), batch_size=cfg.bert_batch, shuffle=shuffle,
                      collate_fn=collate, generator=torch.Generator().manual_seed(seed))


def _autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda")


@torch.no_grad()
def predict_proba(model, loader) -> np.ndarray:
    """Probability of the malicious class."""
    model.eval()
    probs = []
    for ids, mask, _ in loader:
        with _autocast():
            logits = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def _optimizer(model, cfg):
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.bert_weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.bert_lr)


def run_bert(store, cfg, exp, name, seed, train, val, test, text_col="x_bert", keep_model=False):
    """AdamW + linear warm-up/decay, mixed precision on GPU, early stopping on validation F1."""
    if store.done(exp, name, seed):
        return
    set_seed(seed)
    tokenizer = get_tokenizer(cfg.bert_name)

    start = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(cfg.bert_name, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    train_dl = make_loader(train[text_col], train.label.values, tokenizer, cfg, True, seed)
    val_dl = make_loader(val[text_col], val.label.values, tokenizer, cfg, False, seed)
    test_dl = make_loader(test[text_col], test.label.values, tokenizer, cfg, False, seed)

    optimizer = _optimizer(model, cfg)
    total_steps = len(train_dl) * cfg.bert_max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.bert_warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    y_val = val.label.values

    history = defaultdict(list)
    best_f1, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(cfg.bert_max_epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with _autocast():
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            seen += len(y)

        p_val = predict_proba(model, val_dl)
        eps = 1e-7
        val_loss = float(-np.mean(y_val * np.log(p_val + eps) + (1 - y_val) * np.log(1 - p_val + eps)))
        val_f1 = f1_score(y_val, (p_val >= 0.5).astype(int))
        history["train_loss"].append(total_loss / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(((p_val >= 0.5) == y_val).mean()))
        history["val_f1"].append(val_f1)
        print(f"    epoch {epoch + 1}: train loss {total_loss / seen:.4f}  val loss {val_loss:.4f}  val F1 {val_f1:.4f}")

        if val_f1 > best_f1 + 1e-4:
            best_f1, bad_epochs = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch + 1
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.bert_patience:
                break

    model.load_state_dict(best_state)
    train_time = time.time() - start

    start = time.time()
    score = predict_proba(model, test_dl)
    infer_time = time.time() - start

    store.save(exp, name, seed, test, score, (score >= 0.5).astype(int), train_time, infer_time,
               n_params=count_parameters(model), history=dict(history))
    if keep_model:
        path = store.model_path(exp, name, seed)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    del model, optimizer, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
