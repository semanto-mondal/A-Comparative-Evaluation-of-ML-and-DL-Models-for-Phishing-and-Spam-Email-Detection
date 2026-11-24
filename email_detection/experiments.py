"""The experiments of the paper.

E1  in-domain: 7 models x 5 seeds on stratified 80/10/10 splits
E2  cross-source: train on one corpus, test on all emails of the other
E3  ablation: legacy vs new cleaning, 1k vs 50k TF-IDF vocabulary
E4  cost: single-email latency on CPU (and GPU), model size
D   diagnostic: what a model trained on the original (label-confounded) merge learns
"""

import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from .classical import build_classifier, make_tfidf, run_classical
from .config import ALL_MODELS, CLASSICAL_MODELS, SEQUENCE_MODELS
from .data import make_split
from .neural import BiLSTMClassifier, TransformerClassifier, count_parameters, encode, run_sequence_model
from .text import clean_bert, clean_classical, clean_sequence, harmonise


def run_model(store, cfg, exp, name, seed, train, val, test, keep_model=False):
    if name in CLASSICAL_MODELS:
        run_classical(store, cfg, exp, name, seed, train, val, test, keep_model=keep_model)
    elif name in SEQUENCE_MODELS:
        run_sequence_model(store, cfg, exp, name, seed, train, val, test, keep_model=keep_model)
    elif name == "BERT":
        from .bert import run_bert   # imported lazily: transformers is slow to import
        run_bert(store, cfg, exp, name, seed, train, val, test, keep_model=keep_model)
    else:
        raise ValueError(f"unknown model: {name}")


def run_in_domain(store, cfg, data, models=ALL_MODELS):
    """E1. Models from seed 0 are kept for the latency benchmark (E4)."""
    for name in models:
        print(f"\n=== E1 in-domain: {name} ===")
        for seed in cfg.seeds:
            train, val, test = make_split(data, seed, cfg.test_frac, cfg.val_frac)
            run_model(store, cfg, "E1_in_domain", name, seed,
                      data.iloc[train], data.iloc[val], data.iloc[test],
                      keep_model=(seed == cfg.seeds[0]))


def run_cross_source(store, cfg, data, models=ALL_MODELS):
    """E2. The training corpus is split 90/10 into train/validation for each seed."""
    directions = {"E2_Enron_to_IWSPA": ("Enron", "IWSPA"), "E2_IWSPA_to_Enron": ("IWSPA", "Enron")}
    for exp, (source, target) in directions.items():
        src = data[data.source == source]
        tgt = data[data.source == target]
        for name in models:
            print(f"\n=== {exp}: {name} ===")
            for seed in cfg.seeds:
                train_idx, val_idx = train_test_split(np.arange(len(src)), test_size=0.10,
                                                      stratify=src.label.values, random_state=seed)
                run_model(store, cfg, exp, name, seed, src.iloc[train_idx], src.iloc[val_idx], tgt)


def run_ablation(store, cfg, data, models=CLASSICAL_MODELS):
    """E3. Same splits as E1; the third setting (new cleaning, 50k uni+bigrams) is E1 itself."""
    settings = {
        "E3_legacy_clean_1k": ("x_legacy", {"max_features": 1000, "ngrams": (1, 1)}),
        "E3_new_clean_1k": ("x_classical", {"max_features": 1000, "ngrams": (1, 1)}),
    }
    for exp, (text_col, tfidf_kwargs) in settings.items():
        for name in models:
            print(f"\n=== {exp}: {name} ===")
            for seed in cfg.seeds:
                train, val, test = make_split(data, seed, cfg.test_frac, cfg.val_frac)
                run_classical(store, cfg, exp, name, seed, data.iloc[train], data.iloc[val], data.iloc[test],
                              text_col=text_col, tfidf_kwargs=tfidf_kwargs)


def run_latency(store, cfg, data) -> pd.DataFrame:
    """E4. Median time to classify one raw email (cleaning included), batch size 1."""
    seed = cfg.seeds[0]
    _, _, test = make_split(data, seed, cfg.test_frac, cfg.val_frac)
    sample = data.iloc[test].head(cfg.latency_emails)
    emails = (sample.subject + " . " + sample.body).tolist()
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])

    def benchmark(classify):
        classify(emails[0])                                    # warm-up
        times = []
        for email in emails:
            start = time.perf_counter()
            classify(email)
            times.append(time.perf_counter() - start)
        return 1000 * np.median(times), 1000 * np.percentile(times, 95)

    rows = []
    for name in CLASSICAL_MODELS:
        path = store.model_path("E1_in_domain", name, seed, ".pkl")
        try:
            with open(path, "rb") as f:
                vectorizer, clf = pickle.load(f)
        except FileNotFoundError:
            continue
        median, p95 = benchmark(lambda t: clf.predict(vectorizer.transform([clean_classical(harmonise(t))])))
        rows.append({"model": name, "device": "cpu", "latency_ms_median": median, "latency_ms_p95": p95,
                     "model_size_mb": len(pickle.dumps((vectorizer, clf))) / 2**20})

    for name in SEQUENCE_MODELS:
        path = store.model_path("E1_in_domain", name, seed, ".pt")
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except FileNotFoundError:
            continue
        vocab = checkpoint["vocab"]
        for device in devices:
            model = (BiLSTMClassifier(len(vocab) + 2, emb_dim=cfg.seq_emb) if name == "BiLSTM"
                     else TransformerClassifier(len(vocab) + 2, maxlen=cfg.seq_maxlen, d_model=cfg.seq_emb))
            model.load_state_dict(checkpoint["state"])
            model.to(device).eval()

            def classify(t, model=model, device=device):
                x = torch.from_numpy(encode([clean_sequence(harmonise(t))], vocab, cfg.seq_maxlen)).to(device)
                with torch.no_grad():
                    model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            median, p95 = benchmark(classify)
            rows.append({"model": name, "device": device.type, "latency_ms_median": median, "latency_ms_p95": p95,
                         "model_size_mb": count_parameters(model) * 4 / 2**20})

    bert_path = store.model_path("E1_in_domain", "BERT", seed)
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
    except (OSError, ValueError):
        tokenizer = None
    if tokenizer is not None:
        for device in devices:
            model = AutoModelForSequenceClassification.from_pretrained(bert_path).to(device).eval()

            def classify(t, model=model, device=device):
                enc = tokenizer([clean_bert(harmonise(t))], truncation=True, max_length=cfg.bert_maxlen,
                                return_tensors="pt").to(device)
                with torch.no_grad():
                    model(**enc)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            median, p95 = benchmark(classify)
            rows.append({"model": "BERT", "device": device.type, "latency_ms_median": median, "latency_ms_p95": p95,
                         "model_size_mb": count_parameters(model) * 4 / 2**20})
            del model

    latency = pd.DataFrame(rows)
    latency.to_csv(f"{store.root}/tables/latency_raw.csv", index=False)
    return latency


def run_diagnostics(store, cfg, data, pool) -> pd.DataFrame:
    """D1: can the source be told apart from legitimate emails alone?
    D2: a model trained on the original merge (all IWSPA + Enron spam only), tested on Enron legitimate mail.
    """
    rows = []

    legit = data[data.type == "legitimate"]
    for seed in cfg.seeds:
        a, b = train_test_split(np.arange(len(legit)), test_size=0.2, stratify=legit.source.values, random_state=seed)
        vectorizer = make_tfidf(cfg.tfidf_max_features, cfg.tfidf_ngrams)
        X_a = vectorizer.fit_transform(legit.x_classical.iloc[a])
        X_b = vectorizer.transform(legit.x_classical.iloc[b])
        y_a = (legit.source.iloc[a] == "Enron").astype(int)
        y_b = (legit.source.iloc[b] == "Enron").astype(int)
        pred = LogisticRegression(max_iter=5000, solver="liblinear").fit(X_a, y_a).predict(X_b)
        balanced_acc = (recall_score(y_b, pred) + recall_score(1 - y_b, 1 - pred)) / 2
        rows.append({"diagnostic": "D1 source identification (legitimate only)", "model": "LR", "seed": seed,
                     "metric": "balanced accuracy", "value": balanced_acc})

    pool = pool.copy()
    pool["x_classical"] = pool.text.map(clean_classical)
    iwspa_all = pool[pool.source == "IWSPA"]
    enron_spam = pool[(pool.source == "Enron") & (pool.type == "spam")]
    enron_legit = pool[(pool.source == "Enron") & (pool.type == "legitimate")]

    for seed in cfg.seeds:
        rng = np.random.RandomState(seed)
        original_merge = pd.concat([iwspa_all, enron_spam.sample(n=min(7373, len(enron_spam)), random_state=rng)])
        unseen_legit = enron_legit.sample(n=min(2000, len(enron_legit)), random_state=rng)
        tr, te = train_test_split(np.arange(len(original_merge)), test_size=0.2,
                                  stratify=original_merge.label.values, random_state=seed)
        for name in ["LR", "RF"]:
            vectorizer = make_tfidf(cfg.tfidf_max_features, cfg.tfidf_ngrams)
            clf = build_classifier(name, seed).fit(vectorizer.fit_transform(original_merge.x_classical.iloc[tr]),
                                                   original_merge.label.iloc[tr])
            in_dist = accuracy_score(original_merge.label.iloc[te],
                                     clf.predict(vectorizer.transform(original_merge.x_classical.iloc[te])))
            flagged = clf.predict(vectorizer.transform(unseen_legit.x_classical)).mean()
            rows.append({"diagnostic": "D2 original merge: in-distribution test", "model": name, "seed": seed,
                         "metric": "accuracy", "value": in_dist})
            rows.append({"diagnostic": "D2 original merge: Enron legitimate flagged as malicious", "model": name,
                         "seed": seed, "metric": "false-positive rate", "value": flagged})

    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(f"{store.root}/tables/diagnostics_raw.csv", index=False)
    return diagnostics
