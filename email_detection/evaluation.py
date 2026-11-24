"""Metrics and storage of run results.

Each run (experiment x model x seed) writes its test predictions to
results/preds/<exp>__<model>__seed<k>.csv plus a small .json with timing and
size. All tables are computed later from these files, and a run whose file
already exists is skipped, so an interrupted run can simply be restarted.
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y, pred, score, types=None, sources=None) -> dict:
    """Overall metrics (malicious = positive class) plus per-type recall and per-source F1."""
    y, pred, score = np.asarray(y), np.asarray(pred), np.asarray(score)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "fpr": fp / max(tn + fp, 1),        # legitimate emails flagged as malicious
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    if types is not None:
        types = np.asarray(types)
        for kind in ["phishing", "spam"]:
            mask = types == kind
            if mask.any():
                m[f"recall_{kind}"] = (pred[mask] == 1).mean()
    if sources is not None:
        sources = np.asarray(sources)
        for source in ["IWSPA", "Enron"]:
            mask = sources == source
            if mask.any() and len(np.unique(y[mask])) > 1:
                m[f"acc_{source}"] = accuracy_score(y[mask], pred[mask])
                m[f"f1_{source}"] = f1_score(y[mask], pred[mask], zero_division=0)
    return m


class RunStore:
    """Reads and writes per-run predictions under <results_dir>."""

    def __init__(self, results_dir: str):
        self.root = results_dir
        for sub in ["preds", "history", "models", "tables", "figures", "errors"]:
            os.makedirs(os.path.join(results_dir, sub), exist_ok=True)

    def pred_path(self, exp, model, seed):
        return os.path.join(self.root, "preds", f"{exp}__{model}__seed{seed}.csv")

    def model_path(self, exp, model, seed, ext=""):
        return os.path.join(self.root, "models", f"{exp}__{model}__seed{seed}{ext}")

    def history_path(self, exp, model, seed):
        return os.path.join(self.root, "history", f"{exp}__{model}__seed{seed}.json")

    def done(self, exp, model, seed) -> bool:
        return os.path.exists(self.pred_path(exp, model, seed))

    def save(self, exp, model, seed, test_df, score, pred, train_time, infer_time,
             n_params=None, history=None, extra=None):
        out = pd.DataFrame({
            "row": test_df.index.values,
            "label": test_df.label.values,
            "type": test_df.type.values,
            "source": test_df.source.values,
            "score": score,
            "pred": pred,
        })
        out.to_csv(self.pred_path(exp, model, seed), index=False)

        meta = {
            "exp": exp, "model": model, "seed": seed,
            "train_time_s": train_time,
            "infer_time_s": infer_time,
            "infer_ms_per_email": 1000 * infer_time / max(len(test_df), 1),
            "n_params": n_params,
            "n_test": len(test_df),
            **(extra or {}),
        }
        with open(self.pred_path(exp, model, seed).replace(".csv", ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, default=float, indent=1)

        if history is not None:
            with open(self.history_path(exp, model, seed), "w", encoding="utf-8") as f:
                json.dump(history, f, default=float, indent=1)

        m = compute_metrics(out.label, out.pred, out.score, out.type, out.source)
        print(f"  {exp:<20} {model:<12} seed {seed}: acc {m['accuracy']:.4f}  "
              f"f1 {m['f1']:.4f}  auc {m['auc']:.4f}  ({train_time:.0f}s)")

    def load_predictions(self, exp, model, seed) -> pd.DataFrame:
        return pd.read_csv(self.pred_path(exp, model, seed))

    def load_all_runs(self) -> pd.DataFrame:
        """One row of metrics per finished run."""
        rows = []
        pred_dir = os.path.join(self.root, "preds")
        for name in sorted(os.listdir(pred_dir)):
            if not name.endswith(".csv"):
                continue
            exp, model, seed = name[:-4].split("__")
            p = pd.read_csv(os.path.join(pred_dir, name))
            with open(os.path.join(pred_dir, name[:-4] + ".json"), encoding="utf-8") as f:
                meta = json.load(f)
            m = compute_metrics(p.label, p.pred, p.score, p.type, p.source)
            m.update(exp=exp, model=model, seed=int(seed.replace("seed", "")),
                     train_time_s=meta["train_time_s"], infer_ms_per_email=meta["infer_ms_per_email"],
                     n_params=meta.get("n_params"))
            rows.append(m)
        return pd.DataFrame(rows)
