"""Exploratory analysis on the final dataset: VADER sentiment and LSA topics per email type.

Usage:
    python run_eda.py

Writes results/tables/eda_sentiment.csv and results/tables/eda_lsa_topics.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from email_detection.config import Config
from email_detection.data import load_processed

TYPES = ["legitimate", "phishing", "spam"]


def sentiment_by_type(data: pd.DataFrame) -> pd.DataFrame:
    """Share of positive / neutral / negative emails (compound score thresholds +-0.05)."""
    analyzer = SentimentIntensityAnalyzer()
    compound = data.text.str[:5000].map(lambda t: analyzer.polarity_scores(t)["compound"])
    return pd.DataFrame({
        "positive": (compound > 0.05).groupby(data.type).mean(),
        "neutral": compound.between(-0.05, 0.05).groupby(data.type).mean(),
        "negative": (compound < -0.05).groupby(data.type).mean(),
        "median_compound": compound.groupby(data.type).median(),
    }).loc[TYPES]


def lsa_topics(data: pd.DataFrame, n_topics=5, n_terms=8) -> pd.DataFrame:
    """Truncated SVD on a TF-IDF matrix, separately for each email type."""
    rows = []
    for kind in TYPES:
        texts = data[data.type == kind].x_classical
        vectorizer = TfidfVectorizer(max_features=2000, max_df=0.5, min_df=3)
        X = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=n_topics, random_state=0).fit(X)
        terms = vectorizer.get_feature_names_out()
        for i, component in enumerate(svd.components_):
            top = terms[np.argsort(component)[::-1][:n_terms]]
            rows.append({"type": kind, "topic": i + 1, "top_terms": " ".join(top)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=Config.processed_dir)
    parser.add_argument("--results", default=Config.results_dir)
    args = parser.parse_args()

    data = load_processed(Config(processed_dir=args.data))
    out = os.path.join(args.results, "tables")
    os.makedirs(out, exist_ok=True)

    sentiment = sentiment_by_type(data)
    sentiment.to_csv(os.path.join(out, "eda_sentiment.csv"))
    print("Sentiment by type\n", sentiment.round(3), "\n")

    topics = lsa_topics(data)
    topics.to_csv(os.path.join(out, "eda_lsa_topics.csv"), index=False)
    print("LSA topics\n", topics.to_string(index=False))


if __name__ == "__main__":
    main()
