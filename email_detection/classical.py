"""TF-IDF features with Decision Tree, Random Forest, Logistic Regression and linear SVM."""

import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .evaluation import set_seed


def make_tfidf(max_features=50000, ngrams=(1, 2)) -> TfidfVectorizer:
    """Uni+bigrams; drop terms in < 2 emails or > 95% of emails; keep the most frequent."""
    return TfidfVectorizer(max_features=max_features, ngram_range=ngrams, min_df=2, max_df=0.95,
                           sublinear_tf=True, dtype=np.float32)


def build_classifier(name: str, seed: int, C: float = 1.0, rf_trees: int = 300):
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=rf_trees, n_jobs=-1, random_state=seed)
    if name == "LR":
        return LogisticRegression(C=C, max_iter=5000, solver="liblinear", random_state=seed)
    if name == "SVM":
        return LinearSVC(C=C, random_state=seed, max_iter=10000)
    raise ValueError(f"unknown classical model: {name}")


def scores_of(clf, X):
    """Probability for LR/DT/RF, distance to the hyperplane for SVM (both fine for AUC)."""
    return clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)


def count_parameters(clf, name: str) -> int:
    if hasattr(clf, "coef_"):
        return int(clf.coef_.size + clf.intercept_.size)
    if name == "DT":
        return int(clf.tree_.node_count)
    return int(sum(tree.tree_.node_count for tree in clf.estimators_))  # RF: total tree nodes


def run_classical(store, cfg, exp, name, seed, train, val, test,
                  text_col="x_classical", tfidf_kwargs=None, keep_model=False):
    if store.done(exp, name, seed):
        return
    set_seed(seed)
    tfidf_kwargs = tfidf_kwargs or {"max_features": cfg.tfidf_max_features, "ngrams": cfg.tfidf_ngrams}

    start = time.time()
    vectorizer = make_tfidf(**tfidf_kwargs)
    X_train = vectorizer.fit_transform(train[text_col])     # fitted on the training split only
    X_val = vectorizer.transform(val[text_col])

    best_C = None
    if name in ("LR", "SVM"):
        best_f1 = -1.0
        for C in cfg.c_grid:
            candidate = build_classifier(name, seed, C).fit(X_train, train.label)
            f1 = f1_score(val.label, candidate.predict(X_val))
            if f1 > best_f1:
                best_f1, best_C = f1, C

    clf = build_classifier(name, seed, best_C or 1.0, cfg.rf_trees).fit(X_train, train.label)
    train_time = time.time() - start

    start = time.time()
    X_test = vectorizer.transform(test[text_col])
    score, pred = scores_of(clf, X_test), clf.predict(X_test)
    infer_time = time.time() - start

    store.save(exp, name, seed, test, score, pred, train_time, infer_time,
               n_params=count_parameters(clf, name),
               extra={"best_C": best_C, "vocab_size": len(vectorizer.vocabulary_)})

    if keep_model:
        with open(store.model_path(exp, name, seed, ".pkl"), "wb") as f:
            pickle.dump((vectorizer, clf), f)
