"""Paper figures: confusion matrices by email type and learning curves (IEEE column sizes)."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
COLUMN_WIDTH, PAGE_WIDTH = 3.5, 7.16   # inches, IEEE two-column layout

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, store, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(store.root, "figures", f"{name}.{ext}"), dpi=300, bbox_inches="tight",
                    pad_inches=0.01)
    plt.close(fig)


def typed_confusion(store, model, seeds, exp="E1_in_domain") -> np.ndarray:
    """Rows: true legitimate / phishing / spam. Columns: predicted legitimate / malicious. Summed over seeds."""
    matrix = np.zeros((3, 2))
    for seed in seeds:
        p = store.load_predictions(exp, model, seed)
        for i, kind in enumerate(["legitimate", "phishing", "spam"]):
            rows = p[p.type == kind]
            matrix[i] += [(rows.pred == 0).sum(), (rows.pred == 1).sum()]
    return matrix


def plot_confusion(store, seeds, models=("SVM", "BERT"), name="confusion"):
    fig, axes = plt.subplots(1, len(models), figsize=(COLUMN_WIDTH, 1.55), gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)
    for k, (ax, model) in enumerate(zip(axes, models)):
        counts = typed_confusion(store, model, seeds)
        share = counts / counts.sum(1, keepdims=True)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(2):
                ax.text(j, i, f"{share[i, j] * 100:.1f}%\n({int(counts[i, j])})", ha="center", va="center",
                        fontsize=6.3, linespacing=1.0, color="white" if share[i, j] > 0.6 else "#222")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Predicted\nlegitimate", "Predicted\nmalicious"], fontsize=6.3)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Legitimate", "Phishing", "Spam"] if k == 0 else [])
        ax.tick_params(length=0)
        ax.set_title(model)
        for spine in ax.spines.values():
            spine.set_visible(False)
    _save(fig, store, name)


def plot_learning_curves(store, seeds, models=("BiLSTM", "Transformer", "BERT"), name="learning_curves"):
    """Training and validation loss, mean +- std over seeds; dashed line = median best epoch."""
    fig, axes = plt.subplots(1, len(models), figsize=(PAGE_WIDTH, 1.55), gridspec_kw={"wspace": 0.25})
    for ax, model in zip(axes, models):
        histories = []
        for seed in seeds:
            path = store.history_path("E1_in_domain", model, seed)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    histories.append(json.load(f))
        if not histories:
            ax.axis("off")
            continue

        n_epochs = max(len(h["train_loss"]) for h in histories)
        epochs = np.arange(1, n_epochs + 1)
        for key, color, label in [("train_loss", BLUE, "Training"), ("val_loss", ORANGE, "Validation")]:
            values = np.full((len(histories), n_epochs), np.nan)
            for i, h in enumerate(histories):
                values[i, :len(h[key])] = h[key]
            mean, std = np.nanmean(values, 0), np.nanstd(values, 0)
            ax.plot(epochs, mean, color=color, lw=1.6, marker="o", ms=2.8, label=label)
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15, lw=0)

        best = int(np.median([h["best_epoch"] for h in histories]))
        ax.axvline(best, color="#777", lw=0.8, ls="--")
        ax.set_title(f"{model} (median best epoch: {best})", fontsize=7.5)
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.grid(color="#e8e8e8", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[-1].legend(frameon=False, loc="upper right")
    _save(fig, store, name)
