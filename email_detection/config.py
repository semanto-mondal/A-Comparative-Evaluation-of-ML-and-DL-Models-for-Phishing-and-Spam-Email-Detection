"""All settings used in the paper, in one place.

The default values reproduce the reported results. Command-line flags in the
scripts can override a few of them (paths, seeds, fast BERT, subsampling).
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Paths
    iwspa_csv: str = "data/IWSPA-AP-Parsed.csv"
    enron_csv: str = "data/enron_spam_dataset.csv"
    processed_dir: str = "data/processed"
    results_dir: str = "results"

    # Protocol: stratified 80/10/10 train/validation/test, repeated for 5 seeds
    seeds: tuple = (0, 1, 2, 3, 4)
    test_frac: float = 0.10
    val_frac: float = 0.10

    # Dataset construction
    near_dup_threshold: float = 0.90   # MinHash Jaccard threshold
    minhash_perm: int = 128
    iwspa_legit_per_phish: float = 1.0  # legitimate:phishing ratio sampled from IWSPA
    enron_per_class: int = 8000         # legitimate and spam emails sampled from Enron (each)
    sampling_seed: int = 2026           # the dataset itself is identical for every seed
    subsample: int = 0                  # > 0 only for quick smoke tests

    # Classical models (TF-IDF)
    tfidf_max_features: int = 50000
    tfidf_ngrams: tuple = (1, 2)
    c_grid: tuple = (0.1, 1.0, 10.0)    # chosen per seed on the validation split (LR, SVM)
    rf_trees: int = 300

    # Sequence models (BiLSTM, Transformer)
    seq_vocab: int = 60000
    seq_maxlen: int = 250
    seq_emb: int = 128
    seq_batch: int = 64
    seq_lr: float = 1e-3
    seq_max_epochs: int = 10
    seq_patience: int = 2

    # BERT
    bert_name: str = "bert-base-uncased"
    bert_maxlen: int = 256
    bert_batch: int = 16
    bert_lr: float = 2e-5
    bert_max_epochs: int = 4
    bert_patience: int = 1
    bert_warmup: float = 0.1
    bert_weight_decay: float = 0.01

    # Cost benchmark
    latency_emails: int = 200

    def use_fast_bert(self):
        """DistilBERT with 128 tokens: about 3x faster, slightly less accurate."""
        self.bert_name = "distilbert-base-uncased"
        self.bert_maxlen = 128


CLASSICAL_MODELS = ["DT", "RF", "LR", "SVM"]
SEQUENCE_MODELS = ["BiLSTM", "Transformer"]
ALL_MODELS = CLASSICAL_MODELS + SEQUENCE_MODELS + ["BERT"]
