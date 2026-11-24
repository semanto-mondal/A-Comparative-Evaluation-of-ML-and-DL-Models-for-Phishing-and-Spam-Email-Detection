"""Dataset construction: loading, de-duplication, source-balanced sampling and splits."""

import time

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import train_test_split

from .text import clean_bert, clean_classical, clean_legacy, clean_sequence, dedup_key, harmonise


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_corpora(cfg) -> pd.DataFrame:
    """Read both corpora into one table with standardised labels.

    label  : 0 = legitimate, 1 = malicious (phishing or spam)
    type   : legitimate / phishing / spam   (for analysis only)
    source : IWSPA / Enron                  (for analysis only)
    """
    iwspa = pd.read_csv(cfg.iwspa_csv)
    enron = pd.read_csv(cfg.enron_csv)

    iw = pd.DataFrame({
        "subject": iwspa["email_subject"].fillna("").astype(str),
        "body": iwspa["email_body"].fillna("").astype(str),
        "label": iwspa["is_phishing"].astype(int),
        "source": "IWSPA",
    })
    iw["type"] = np.where(iw.label == 1, "phishing", "legitimate")

    en = pd.DataFrame({
        "subject": enron["email_subject"].fillna("").astype(str),
        "body": enron["email_body"].fillna("").astype(str),
        "label": enron["is_spam"].astype(int),
        "source": "Enron",
    })
    en["type"] = np.where(en.label == 1, "spam", "legitimate")

    emails = pd.concat([iw, en], ignore_index=True)
    emails["raw_id"] = np.arange(len(emails))      # row number in IWSPA followed by Enron
    not_empty = (emails.subject.str.strip() + emails.body.str.strip()).str.len() > 0
    return emails[not_empty].reset_index(drop=True)


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------

class Audit:
    """Counts per source and type after each preparation step (Table I in the paper)."""

    def __init__(self):
        self.rows = []

    def log(self, step: str, df: pd.DataFrame):
        row = {"step": step, "total": len(df)}
        for (source, kind), n in df.groupby(["source", "type"]).size().items():
            row[f"{source}-{kind}"] = int(n)
        self.rows.append(row)
        print(f"{step:<40} {len(df):>6}")

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).fillna(0).astype({c: int for c in self._count_columns()})

    def _count_columns(self):
        return [c for c in pd.DataFrame(self.rows).columns if c != "step"]


def _shingles(key: str, k: int = 3, max_words: int = 600) -> set:
    words = key.split()[:max_words]
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def near_duplicate_groups(keys, threshold: float, num_perm: int) -> np.ndarray:
    """Group emails whose word 3-shingles have Jaccard similarity >= threshold.

    MinHash-LSH finds candidate pairs; union-find merges them into groups
    (so A ~ B and B ~ C puts A, B and C in one group).
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures = []
    for i, key in enumerate(keys):
        m = MinHash(num_perm=num_perm, seed=1)
        m.update_batch([s.encode("utf8") for s in _shingles(key)])
        lsh.insert(str(i), m)
        signatures.append(m)

    parent = list(range(len(keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, m in enumerate(signatures):
        for j in map(int, lsh.query(m)):
            if j != i:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    return np.array([find(i) for i in range(len(keys))])


def deduplicate(emails: pd.DataFrame, cfg, audit: Audit) -> pd.DataFrame:
    """Remove label conflicts, exact duplicates and near-duplicates (in that order)."""
    labels_per_key = emails.groupby("key").label.nunique()
    conflicting = set(labels_per_key[labels_per_key > 1].index)
    df = emails[~emails.key.isin(conflicting)]
    audit.log("Exact-label conflicts removed", df)

    df = df.drop_duplicates("key").reset_index(drop=True)
    audit.log("Exact duplicates removed", df)

    start = time.time()
    df["group"] = near_duplicate_groups(df.key.values, cfg.near_dup_threshold, cfg.minhash_perm)
    print(f"  MinHash-LSH: {time.time() - start:.0f}s, "
          f"{int((df.group.value_counts() > 1).sum())} groups with more than one email")

    labels_per_group = df.groupby("group").label.nunique()
    df = df[~df.group.isin(labels_per_group[labels_per_group > 1].index)]
    audit.log("Near-duplicate label conflicts removed", df)

    df = df.drop_duplicates("group").reset_index(drop=True)
    audit.log("Near-duplicates removed", df)
    return df


# ---------------------------------------------------------------------------
# Source-balanced dataset
# ---------------------------------------------------------------------------

def source_balanced_sample(pool: pd.DataFrame, cfg) -> pd.DataFrame:
    """Balance classes *within* each corpus, so the source carries no label information.

    IWSPA: all phishing + the same number of legitimate emails.
    Enron: cfg.enron_per_class legitimate + cfg.enron_per_class spam.
    """
    rng = np.random.RandomState(cfg.sampling_seed)

    def take(subset, n):
        return subset.sample(n=min(n, len(subset)), random_state=rng)

    def part(source, kind):
        return pool[(pool.source == source) & (pool.type == kind)]

    iw_phish = part("IWSPA", "phishing")
    iw_legit = take(part("IWSPA", "legitimate"), int(round(len(iw_phish) * cfg.iwspa_legit_per_phish)))
    en_spam = take(part("Enron", "spam"), cfg.enron_per_class)
    en_legit = take(part("Enron", "legitimate"), cfg.enron_per_class)

    data = pd.concat([iw_phish, iw_legit, en_spam, en_legit])
    data = data.sample(frac=1, random_state=rng).reset_index(drop=True)

    if cfg.subsample:
        per_group = cfg.subsample // 4
        data = pd.concat([g.sample(n=min(len(g), per_group), random_state=0)
                          for _, g in data.groupby(["source", "type"])]).reset_index(drop=True)

    data["stratum"] = data.source + "-" + data.type
    return data


def add_text_views(data: pd.DataFrame) -> pd.DataFrame:
    """One input text per model family (see text.py)."""
    data["x_classical"] = data.text.map(clean_classical)
    data["x_seq"] = data.text.map(clean_sequence)
    data["x_bert"] = data.text.map(clean_bert)
    data["x_legacy"] = (data.subject + " " + data.body).map(clean_legacy)  # legacy cleaning used raw text
    return data


def dataset_from_ids(emails: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a dataset from a list of raw_ids (in the given order).

    Used to recreate the exact paper dataset: near-duplicate candidates from
    MinHash-LSH can differ slightly between library versions, which changes
    the sampled emails, so the paper's selection is released as an ID list.
    """
    data = emails.set_index("raw_id").loc[ids.raw_id.values].reset_index()
    mismatch = (data.type.values != ids.type.values) | (data.source.values != ids.source.values)
    if mismatch.any():
        raise ValueError(f"{int(mismatch.sum())} IDs do not match the expected source/type; "
                         "are these the original IWSPA and Enron CSV files?")
    data["group"] = np.arange(len(data))          # near-duplicates were removed when the list was made
    data["stratum"] = data.source + "-" + data.type
    return data


def build_dataset(cfg, ids_path=None):
    """Full preparation pipeline. Returns (final dataset, de-duplicated pool, audit table).

    With ids_path, the final dataset is the released paper selection instead of a new sample.
    """
    audit = Audit()

    emails = load_corpora(cfg)
    audit.log("Raw (non-empty)", emails)

    start = time.time()
    emails["text"] = (emails.subject + " . " + emails.body).map(harmonise)
    emails["key"] = emails.text.map(dedup_key)
    emails = emails[emails.key.str.len() > 0].reset_index(drop=True)
    print(f"  Harmonised {len(emails)} emails in {time.time() - start:.0f}s")

    pool = deduplicate(emails, cfg, audit)

    if ids_path:
        data = dataset_from_ids(emails, pd.read_csv(ids_path))
        audit.log("Final source-balanced dataset (paper IDs)", data)
    else:
        data = source_balanced_sample(pool, cfg)
        audit.log("Final source-balanced dataset", data)

    data = add_text_views(data)
    return data, pool, audit.table()


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def make_split(data: pd.DataFrame, seed: int, test_frac: float, val_frac: float):
    """Stratified (source x type) train/validation/test indices, with leakage checks."""
    idx = np.arange(len(data))
    strata = data.stratum.values

    train_val, test = train_test_split(idx, test_size=test_frac, stratify=strata, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_frac / (1 - test_frac),
                                  stratify=strata[train_val], random_state=seed)

    for a, b in [(train, val), (train, test), (val, test)]:
        assert not set(a) & set(b), "an email appears in two splits"
        assert not set(data.key.values[a]) & set(data.key.values[b]), "identical text in two splits"
        assert not set(data.group.values[a]) & set(data.group.values[b]), "near-duplicate group in two splits"
    return train, val, test


def load_processed(cfg) -> pd.DataFrame:
    """Read the dataset written by prepare_data.py (empty strings stay empty strings)."""
    data = pd.read_csv(f"{cfg.processed_dir}/final_dataset.csv", keep_default_na=False)
    data["label"] = data.label.astype(int)
    data["group"] = data.group.astype(int)
    return data
