import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, auc
from TrainingTools import *

def plot_validation_roc(model, val_loader, device="cpu", output_path="roc.png", channel=""):
    model.to(device)
    model.eval()

    all_scores = []
    all_labels = []
    all_weights = []

    with torch.no_grad():
        for features, disco, labels, weights in val_loader:
            features = features.to(device)

            logits = model(features)

            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)

            scores = torch.sigmoid(logits[:, 0])

            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy().ravel())
            all_weights.append(weights.cpu().numpy().ravel())

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)
    sample_weight = np.concatenate(all_weights)

    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    roc_auc = roc_auc_score(y_true,y_score,sample_weight=sample_weight)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curve")
    plt.text(0.98, 0.16, channel, ha="right", transform=plt.gca().transAxes)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"ROC AUC = {roc_auc:.6f}")
    print(f"Saved ROC curve to {output_path}")

    return roc_auc

def plot_score_densities(
    model,
    data_loader,
    device="cpu",
    output_path="plots",
    normalize=True
):
    model.to(device)
    model.eval()

    all_scores = []
    all_labels = []
    all_weights = []
    all_disco = []
    with torch.no_grad():
        for features, disco, labels, weights in data_loader:
            features = features.to(device)

            logits = model(features)

            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)

            scores = torch.sigmoid(logits).cpu().numpy()

            all_scores.append(scores)
            all_labels.append(labels.cpu().numpy().ravel())
            all_weights.append(weights.cpu().numpy().ravel())
            all_disco.append(disco.cpu().numpy().ravel())

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels)
    weights = np.concatenate(all_weights)
    disco = np.concatenate(all_disco)

    sig_mask = labels == 1
    bkg_mask = labels == 0

    score_cols = (
        ["dnn_score"]
    )

    fig, axes = plt.subplots(
        1,
        len(score_cols),
        figsize=(8 * len(score_cols), 6),
        squeeze=False,
    )

    for idx, col in enumerate(score_cols):
        ax = axes[0, idx]

        values = scores[:, idx]

        ax.hist(
            values[sig_mask],
            bins=50,
            range=(0, 1),
            weights=weights[sig_mask],
            density=normalize,
            histtype="step",
            linewidth=2,
            label="Signal",
            color="tab:red",
        )

        ax.hist(
            values[bkg_mask],
            bins=50,
            range=(0, 1),
            weights=weights[bkg_mask],
            density=normalize,
            histtype="step",
            linewidth=2,
            label="Background",
            color="tab:blue",
        )

        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.set_title(f"Score density: {col}")
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(f'{output_path}/score_density.png', dpi=200)
    plt.close()

    print(f"Saved score_density.png to {output_path}/")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        squeeze=False,
    )

    for idx, col in enumerate(score_cols):
        values = scores[:, idx]
        # Signal
        ax = axes[0, 0]
        h = ax.hist2d(
            values[sig_mask],
            disco[sig_mask],
            bins=(50, 50),
            range=((0, 1), (np.min(disco), np.max(disco))),
            weights=weights[sig_mask],
            cmap="Reds",
        )
        fig.colorbar(h[3], ax=ax, label="Weighted counts")
        ax.set_xlabel(col)
        ax.set_ylabel("VBS BDT score")
        ax.set_title("Signal")

        # Background
        ax = axes[0, 1]
        h = ax.hist2d(
            values[bkg_mask],
            disco[bkg_mask],
            bins=(50, 50),
            range=((0, 1), (np.min(disco), np.max(disco))),
            weights=weights[bkg_mask],
            cmap="Blues",
        )
        fig.colorbar(h[3], ax=ax, label="Weighted counts")
        ax.set_xlabel(col)
        ax.set_ylabel("VBS BDT score")
        ax.set_title("Background")

    plt.tight_layout()

    plt.savefig(f'{output_path}/score_vs_disco.png', dpi=200)
    plt.close()

    print(f"Saved score_vs_disco.png to {output_path}/")
    
    score = scores[:, 0]
    score_bins = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
   
    for lo, hi in score_bins:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Signal
        sel = sig_mask & (score >= lo) & (score < hi)

        ax.hist(
                disco[sel],
                bins=50,
                weights=weights[sel],
                density=True,
                histtype="step",
                linewidth=2,
                color="tab:red",
                label=f"{lo:.1f} ≤ score < {hi:.1f}",
            )
        
        # Background
        sel = bkg_mask & (score >= lo) & (score < hi)
        ax.hist(
            disco[sel],
            bins=50,
            weights=weights[sel],
            density=True,
            histtype="step",
            linewidth=2,
            color="tab:blue",
            label="Background",
        )
        ax.set_xlabel("disco")
        ax.set_ylabel("Density")
        ax.set_title(f"{lo:.1f} ≤ DNN score < {hi:.1f}")
        ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{output_path}/disco_score_{lo:.1f}_{hi:.1f}.png", dpi=200)
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, choices=[2, 3], required=True)
    parser.add_argument("--nFJ", type=int, choices=[1, 2], required=True)
    parser.add_argument("--dataset", required=True, help="Dated dataset dir written by main.py, e.g. 260916_dataset")
    parser.add_argument("--signal", default="c2v1p5_c3_1p0", choices=["c2v1p0_c3_1p0", "c2v1p0_c3_10p0", "c2v1p5_c3_1p0"])
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    tag = f"run{args.run}_2L_{args.nFJ}FJ"
    plot_dir = base / "plots" / tag
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dataset = torch.load(base / args.dataset / args.signal / f"{tag}_valid.pt", map_location="cpu", weights_only=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, shuffle=False, pin_memory=(device == "cuda"), num_workers=4)

    model = ABCDModel(
        input_size=val_dataset[0][0].shape[0],
        hidden_layers=[64, 32, 16],
        learning_rate=0.001,
        bce_weight=1.0,
        disco_lambda=10.0,
        flavor='single',
        use_batchnorm=True,
        dropout=0.2,
        weight_decay=0.01,
        label_smoothing=0.0,
        use_lr_scheduler=True,
        lr_scheduler_patience=5,
        lr_scheduler_factor=0.5,
        lr_scheduler_min_lr=1e-6,
    )
    model.load_state_dict(torch.load(base / "models" / f"{tag}_best_model.pt", map_location="cpu"))

    plot_validation_roc(
        model,
        val_loader,
        device=device,
        output_path=plot_dir / "roc.png",
        channel=f"Run {args.run} nFJ={args.nFJ}",
    )

    plot_score_densities(
        model,
        val_loader,
        device=device,
        output_path=plot_dir,
        normalize=True,
    )

if __name__ == "__main__":
    main()
