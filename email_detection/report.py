"""Summary tables (CSV + LaTeX) and significance tests, computed from saved predictions."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from .config import ALL_MODELS

MAIN_METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
TYPE_METRICS = ["recall_phishing", "recall_spam", "fpr", "f1_IWSPA", "f1_Enron"]
LOWER_IS_BETTER = {"fpr", "train_time_s", "infer_ms_per_email"}


def ci95(values) -> float:
    """Half-width of the 95% confidence interval of the mean (t-distribution)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def summarise(runs: pd.DataFrame, exp: str, metrics) -> pd.DataFrame:
    """Mean, std and 95% CI over seeds, one row per model."""
    subset = runs[runs.exp == exp]
    rows = []
    for model in [m for m in ALL_MODELS if m in subset.model.unique()]:
        g = subset[subset.model == model]
        row = {"model": model, "n_seeds": len(g)}
        for k in metrics:
            if k in g:
                row[k] = g[k].mean()
                row[f"{k}_std"] = g[k].std(ddof=1)
                row[f"{k}_ci95"] = ci95(g[k])
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_std(summary: pd.DataFrame, metrics, digits=3) -> pd.DataFrame:
    """'0.986 $\\pm$ 0.003' cells, best value in bold."""
    out = summary[["model"]].copy()
    for k in metrics:
        if k not in summary:
            continue
        best = summary[k].min() if k in LOWER_IS_BETTER else summary[k].max()
        cells = []
        for mean, std in zip(summary[k], summary[f"{k}_std"]):
            value = f"{mean:.{digits}f}"
            if np.isclose(mean, best):
                value = f"\\textbf{{{value}}}"
            cells.append(f"{value} $\\pm$ {std:.{digits}f}")
        out[k] = cells
    return out


def write_latex(df: pd.DataFrame, caption: str, label: str, path: str):
    columns = list(df.columns)
    lines = [
        "\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * (len(columns) - 1) + "}", "\\hline",
        " & ".join(c.replace("_", " ") for c in columns) + " \\\\", "\\hline",
    ]
    lines += [" & ".join(str(v) for v in row) + " \\\\" for row in df.values]
    lines += ["\\hline", "\\end{tabular}}", "\\end{table}"]

    tex = "\n".join(lines)
    for symbol, latex in {"→": "$\\rightarrow$", "–": "--", "±": "$\\pm$", "≥": "$\\geq$"}.items():
        tex = tex.replace(symbol, latex)
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


def cross_source_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    settings = [("E1_in_domain", "In-domain"), ("E2_Enron_to_IWSPA", "Enron→IWSPA"), ("E2_IWSPA_to_Enron", "IWSPA→Enron")]
    for model in ALL_MODELS:
        row = {"model": model}
        for exp, short in settings:
            g = runs[(runs.exp == exp) & (runs.model == model)]
            for k in ["f1", "recall", "fpr"]:
                row[f"{short} {k}"] = g[k].mean() if len(g) else np.nan
                row[f"{short} {k}_std"] = g[k].std(ddof=1) if len(g) > 1 else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    return table[table.drop(columns="model").notna().any(axis=1)].reset_index(drop=True)


def ablation_table(runs: pd.DataFrame, models) -> pd.DataFrame:
    settings = [("E3_legacy_clean_1k", "Legacy cleaning + TF-IDF 1k"),
                ("E3_new_clean_1k", "New cleaning + TF-IDF 1k"),
                ("E1_in_domain", "New cleaning + TF-IDF 50k (1-2-gram)")]
    rows = []
    for exp, label in settings:
        for model in models:
            g = runs[(runs.exp == exp) & (runs.model == model)]
            if len(g):
                rows.append({"setting": label, "model": model, "f1": g.f1.mean(), "f1_std": g.f1.std(ddof=1),
                             "recall_phishing": g.recall_phishing.mean(), "fpr": g.fpr.mean()})
    return pd.DataFrame(rows)


def mcnemar_p(y, a, b) -> float:
    """Exact McNemar test on the emails where the two models disagree."""
    a_right, b_right = a == y, b == y
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    if only_a + only_b == 0:
        return 1.0
    return stats.binomtest(only_a, only_a + only_b, 0.5).pvalue


def significance_table(store, seeds, reference="BERT") -> pd.DataFrame:
    """Paired t-test on F1 across seeds and McNemar's test per seed: reference vs every other model."""
    rows = []
    for exp in ["E1_in_domain", "E2_Enron_to_IWSPA", "E2_IWSPA_to_Enron"]:
        if not store.done(exp, reference, seeds[0]):
            continue
        for model in [m for m in ALL_MODELS if m != reference]:
            f1_ref, f1_other, p_values = [], [], []
            for seed in seeds:
                if not (store.done(exp, reference, seed) and store.done(exp, model, seed)):
                    continue
                a = store.load_predictions(exp, reference, seed)
                b = store.load_predictions(exp, model, seed)
                f1_ref.append(f1_score(a.label, a.pred))
                f1_other.append(f1_score(b.label, b.pred))
                p_values.append(mcnemar_p(a.label.values, a.pred.values, b.pred.values))
            if len(f1_ref) >= 2:
                rows.append({
                    "exp": exp,
                    "comparison": f"{reference} vs {model}",
                    "delta_f1": np.mean(f1_ref) - np.mean(f1_other),
                    "paired_t_p": stats.ttest_rel(f1_ref, f1_other).pvalue,
                    "mcnemar_seeds_p_below_0.05": f"{sum(p < 0.05 for p in p_values)}/{len(p_values)}",
                    "mcnemar_median_p": np.median(p_values),
                })
    return pd.DataFrame(rows)


def cost_table(runs: pd.DataFrame, latency: pd.DataFrame) -> pd.DataFrame:
    e1 = runs[runs.exp == "E1_in_domain"]
    table = summarise(runs, "E1_in_domain", ["train_time_s"])
    table = table.merge(e1.groupby("model").n_params.first().reset_index(), on="model", how="left")
    if latency is not None and len(latency):
        per_device = latency.pivot_table(index="model", columns="device", values="latency_ms_median")
        per_device = per_device.add_prefix("latency_ms_").reset_index()
        size = latency.groupby("model").model_size_mb.first().reset_index()
        table = table.merge(per_device, on="model", how="left").merge(size, on="model", how="left")
    return table
