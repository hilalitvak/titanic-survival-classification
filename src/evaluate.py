import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)


def print_metrics(y_true: np.ndarray, y_probs: np.ndarray):
    preds = (y_probs >= 0.5).astype(int)
    print('\n=== Validation Metrics ===')
    print(f'Accuracy  : {accuracy_score(y_true, preds):.4f}')
    print(f'Precision : {precision_score(y_true, preds):.4f}')
    print(f'Recall    : {recall_score(y_true, preds):.4f}')
    print(f'F1        : {f1_score(y_true, preds):.4f}')
    print(f'ROC-AUC   : {roc_auc_score(y_true, y_probs):.4f}')


def plot_results(y_true: np.ndarray, y_probs: np.ndarray) -> plt.Figure:
    preds = (y_probs >= 0.5).astype(int)
    auc   = roc_auc_score(y_true, y_probs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Model Evaluation', fontsize=14, fontweight='bold', y=1.02)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, preds)
    axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title('Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Died', 'Survived'])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(['Died', 'Survived'])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha='center', va='center',
                         fontsize=16, color='white' if cm[i, j] > cm.max() / 2 else 'black')

    # ── ROC curve ──────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    axes[1].plot(fpr, tpr, lw=2, color='steelblue', label=f'AUC = {auc:.3f}')
    axes[1].plot([0, 1], [0, 1], '--', color='gray', lw=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve', fontweight='bold')
    axes[1].legend()
    axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.02])

    # ── Precision-Recall curve ─────────────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    baseline = y_true.mean()
    axes[2].plot(recall, precision, lw=2, color='darkorange')
    axes[2].axhline(baseline, linestyle='--', color='gray', lw=1, label=f'Baseline ({baseline:.2f})')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision–Recall Curve', fontweight='bold')
    axes[2].legend()
    axes[2].set_xlim([0, 1]); axes[2].set_ylim([0, 1.02])

    plt.tight_layout()
    return fig
