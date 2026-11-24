"""Step 1: build the de-duplicated, source-balanced dataset.

By default the final dataset is the exact selection used in the paper
(data/paper_dataset_ids.csv: row numbers only, no email text). Use --resample
to draw a new source-balanced sample from the de-duplicated pool instead.

Writes to data/processed/:
    final_dataset.csv         the 17,934 emails with labels, type, source and the text views
    deduplicated_pool.csv     all unique emails before sampling (used by the diagnostic)
    dataset_preparation.csv   counts after each step

Usage:
    python prepare_data.py
    python prepare_data.py --iwspa path/to/IWSPA-AP-Parsed.csv --enron path/to/enron_spam_dataset.csv
    python prepare_data.py --resample
"""

import argparse
import os

from email_detection.config import Config
from email_detection.data import build_dataset, make_split


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iwspa", default=Config.iwspa_csv, help="IWSPA-AP 2018 CSV")
    parser.add_argument("--enron", default=Config.enron_csv, help="Enron-Spam CSV")
    parser.add_argument("--out", default=Config.processed_dir, help="output folder")
    parser.add_argument("--ids", default="data/paper_dataset_ids.csv", help="released paper selection")
    parser.add_argument("--resample", action="store_true", help="ignore --ids and sample a new dataset")
    parser.add_argument("--subsample", type=int, default=0, help="keep only N emails (smoke tests)")
    args = parser.parse_args()

    cfg = Config(iwspa_csv=args.iwspa, enron_csv=args.enron, processed_dir=args.out, subsample=args.subsample)
    os.makedirs(cfg.processed_dir, exist_ok=True)

    use_ids = not args.resample and not args.subsample and os.path.exists(args.ids)
    data, pool, audit = build_dataset(cfg, ids_path=args.ids if use_ids else None)

    # Check every seed's split once, so a leakage problem shows up here rather than mid-training
    for seed in cfg.seeds:
        make_split(data, seed, cfg.test_frac, cfg.val_frac)

    data.to_csv(os.path.join(cfg.processed_dir, "final_dataset.csv"), index=False)
    pool.to_csv(os.path.join(cfg.processed_dir, "deduplicated_pool.csv"), index=False)
    audit.to_csv(os.path.join(cfg.processed_dir, "dataset_preparation.csv"), index=False)

    print("\nDataset preparation:")
    print(audit.to_string(index=False))
    if use_ids:
        print("\nFinal dataset = paper selection from", args.ids,
              "\nThe intermediate counts above can differ slightly from Table I with other datasketch/scipy "
              "versions;\nthe paper's counts are in data/paper_dataset_preparation.csv.")
    print(f"\nSaved {len(data)} emails to {cfg.processed_dir}/final_dataset.csv "
          f"(splits checked for {len(cfg.seeds)} seeds: no shared text or near-duplicates)")


if __name__ == "__main__":
    main()
