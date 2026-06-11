import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

def save_plot(model_name, plot_name, project_dir="plots"):
    save_dir = os.path.join(project_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"{plot_name}.png")
    plt.savefig(path, bbox_inces="tight", dpi=300)
    plt.close()

    print(f"Saved: {path}")

def plot_roc(y_true, y_scores, model_name):
    """
    Saves the plot of ROC curve and computes AUC

    Parameters:
        y_true: array of true scores
        y_scores = predicted scores
        model_name: name of model

    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC: {roc_auc:.3f}")
    plt.plot([0, 1], [1, 0], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    save_plot(model_name, "roc")

def plot_pr(y_true, y_scores, model_name):
    """Saves the plot of the Precision-Recall curve.
    
    Parameters:
        Parameters:
        y_true: array of true scores
        y_scores = predicted scores
        model_name: name of model
        """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")

    save_plot(model_name, "pr")

def plot_sensitivity_fp(y_true, y_scores, model_name, n_points=50):
    thresholds = np.linspace(0, 1, n_points)

    sensitivities = []
    fps = []

    for t in thresholds:
        preds = (y_scores >= t).astype(int)

        TP = np.sum((preds == 1) & (y_true == 1))
        FP = np.sum((preds == 1) & (y_true == 0))
        FN = np.sum((preds == 0) & (y_true == 1))

        sensitivity = TP / (TP + FN + 1e-8)
        fp_rate = FP / len(y_true)

        sensitivities.append(sensitivity)
        fps.append(fp_rate)

    plt.figure()
    plt.plot(fps, sensitivities)
    plt.xlabel("False Positives (approx per sample)")
    plt.ylabel("Sensitivity")
    plt.title("FROC-style Curve (approx)")

    save_plot(model_name, "sens_fp")