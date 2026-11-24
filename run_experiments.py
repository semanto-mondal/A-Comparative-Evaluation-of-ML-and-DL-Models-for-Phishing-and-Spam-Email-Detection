"""Step 2: train and evaluate the models.

Every run saves its test predictions under results/preds/. Finished runs are
skipped, so you can stop the script at any time and start it again.

Usage:
    python run_experiments.py                              # everything (about 2-3 h on an RTX 4060)
    python run_experiments.py --experiments E1             # in-domain only
    python run_experiments.py --experiments E1 --models LR SVM BERT
    python run_experiments.py --fast-bert                  # DistilBERT, 128 tokens

Experiments: E1 in-domain, E2 cross-source, E3 ablation, E4 latency, D diagnostic.
"""

import argparse

import pandas as pd

from email_detection.config import ALL_MODELS, CLASSICAL_MODELS, Config
from email_detection.data import load_processed
from email_detection.evaluation import RunStore
from email_detection.experiments import (run_ablation, run_cross_source, run_diagnostics, run_in_domain,
                                         run_latency)
from email_detection.neural import DEVICE


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=["E1", "E2", "E3", "E4", "D"],
                        choices=["E1", "E2", "E3", "E4", "D"])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(Config.seeds))
    parser.add_argument("--data", default=Config.processed_dir, help="folder written by prepare_data.py")
    parser.add_argument("--results", default=Config.results_dir)
    parser.add_argument("--fast-bert", action="store_true", help="use DistilBERT with 128 tokens")
    parser.add_argument("--bert-name", default=None, help="any Hugging Face model name or local path")
    args = parser.parse_args()

    cfg = Config(processed_dir=args.data, results_dir=args.results, seeds=tuple(args.seeds))
    if args.fast_bert:
        cfg.use_fast_bert()
    if args.bert_name:
        cfg.bert_name = args.bert_name

    print(f"Device: {DEVICE} | seeds: {list(cfg.seeds)} | BERT: {cfg.bert_name} ({cfg.bert_maxlen} tokens)")
    data = load_processed(cfg)
    store = RunStore(cfg.results_dir)

    if "E1" in args.experiments:
        run_in_domain(store, cfg, data, args.models)
    if "E2" in args.experiments:
        run_cross_source(store, cfg, data, args.models)
    if "E3" in args.experiments:
        run_ablation(store, cfg, data, [m for m in args.models if m in CLASSICAL_MODELS])
    if "E4" in args.experiments:
        print("\n=== E4 latency (models from E1, seed 0) ===")
        print(run_latency(store, cfg, data).round(2).to_string(index=False))
    if "D" in args.experiments:
        print("\n=== Diagnostic: original label-confounded merge ===")
        pool = pd.read_csv(f"{cfg.processed_dir}/deduplicated_pool.csv", keep_default_na=False)
        pool["label"] = pool.label.astype(int)
        diagnostics = run_diagnostics(store, cfg, data, pool)
        print(diagnostics.groupby(["diagnostic", "model"]).value.agg(["mean", "std"]).round(4))

    print("\nDone. Next: python make_report.py")


if __name__ == "__main__":
    main()
