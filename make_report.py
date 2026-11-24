"""Step 3: tables (CSV + LaTeX), significance tests and figures from the saved predictions.

Usage:
    python make_report.py
    python make_report.py --results results

Writes results/tables/*.csv|tex and results/figures/*.pdf|png.
"""

import argparse
import os

import pandas as pd

from email_detection.config import CLASSICAL_MODELS, Config
from email_detection.evaluation import RunStore
from email_detection.figures import plot_confusion, plot_learning_curves
from email_detection.report import (MAIN_METRICS, TYPE_METRICS, ablation_table, cost_table, cross_source_table,
                                    format_mean_std, significance_table, summarise, write_latex)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=Config.results_dir)
    parser.add_argument("--data", default=Config.processed_dir)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(Config.seeds))
    args = parser.parse_args()

    store = RunStore(args.results)
    tables = os.path.join(args.results, "tables")
    runs = store.load_all_runs()
    runs.to_csv(os.path.join(tables, "all_runs.csv"), index=False)
    pd.set_option("display.width", 200)

    # Table I: dataset preparation
    audit_path = os.path.join(args.data, "dataset_preparation.csv")
    if os.path.exists(audit_path):
        print("Dataset preparation\n", pd.read_csv(audit_path).to_string(index=False), "\n")

    # Table III: in-domain results, overall and by type/source
    in_domain = summarise(runs, "E1_in_domain", MAIN_METRICS + TYPE_METRICS)
    in_domain.to_csv(os.path.join(tables, "in_domain.csv"), index=False)
    write_latex(format_mean_std(in_domain, MAIN_METRICS + TYPE_METRICS),
                "In-domain test results (mean $\\pm$ std over five seeds).", "tab:main",
                os.path.join(tables, "in_domain.tex"))
    print("In-domain (E1)\n", format_mean_std(in_domain, MAIN_METRICS + TYPE_METRICS).to_string(index=False), "\n")

    # Table IV: cross-source
    cross = cross_source_table(runs)
    cross.to_csv(os.path.join(tables, "cross_source.csv"), index=False)
    print("Cross-source (E2)\n", cross.filter(regex="model|f1$|fpr$").round(3).to_string(index=False), "\n")

    # Ablation
    ablation = ablation_table(runs, CLASSICAL_MODELS)
    ablation.to_csv(os.path.join(tables, "ablation.csv"), index=False)
    print("Ablation (E3)\n", ablation.round(4).to_string(index=False), "\n")

    # Table V: cost
    latency_path = os.path.join(tables, "latency_raw.csv")
    latency = pd.read_csv(latency_path) if os.path.exists(latency_path) else None
    cost = cost_table(runs, latency)
    cost.to_csv(os.path.join(tables, "cost.csv"), index=False)
    print("Cost (E4)\n", cost.round(2).to_string(index=False), "\n")

    # Significance
    significance = significance_table(store, args.seeds)
    significance.to_csv(os.path.join(tables, "significance.csv"), index=False)
    print("Significance (BERT vs others)\n", significance.round(4).to_string(index=False), "\n")

    # Diagnostic
    diag_path = os.path.join(tables, "diagnostics_raw.csv")
    if os.path.exists(diag_path):
        diagnostics = pd.read_csv(diag_path).groupby(["diagnostic", "model", "metric"]).value.agg(["mean", "std"])
        diagnostics.reset_index().to_csv(os.path.join(tables, "diagnostics.csv"), index=False)
        print("Diagnostic\n", diagnostics.round(4), "\n")

    # Figures
    e1_models = set(runs[runs.exp == "E1_in_domain"].model)
    if {"SVM", "BERT"} <= e1_models:
        plot_confusion(store, args.seeds)
    plot_learning_curves(store, args.seeds)
    print(f"Tables in {tables}/, figures in {os.path.join(args.results, 'figures')}/")


if __name__ == "__main__":
    main()
