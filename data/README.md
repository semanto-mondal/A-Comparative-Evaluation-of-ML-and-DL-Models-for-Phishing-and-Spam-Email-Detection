# Data

The two corpora are not redistributed here. Please obtain them from their original sources and respect their terms of use.

| File | Corpus | Required columns |
|---|---|---|
| `IWSPA-AP-Parsed.csv` | IWSPA-AP 2018 anti-phishing shared task (El Aassal et al., 2018) | `email_subject`, `email_body`, `is_phishing` (1 = phishing) |
| `enron_spam_dataset.csv` | Enron-Spam (Metsis et al., 2006) | `email_subject`, `email_body`, `is_spam` (1 = spam) |

Other columns (for example `email_from`, `email_to`) are ignored.

Expected raw sizes (non-empty emails):

| | Legitimate | Malicious |
|---|---|---|
| IWSPA-AP 2018 | 9,163 | 1,131 phishing |
| Enron-Spam | 16,545 | 17,119 spam |

If your files have different names or locations:

```bash
python prepare_data.py --iwspa path/to/iwspa.csv --enron path/to/enron.csv
```

`prepare_data.py` writes the processed dataset to `data/processed/`.
