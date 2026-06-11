import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
from tensorflow import keras

def save_plot(family_name, plot_name, project_dir="plots"):
    save_dir = os.path.join(project_dir, family_name)
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, f"{plot_name}.png")
    plt.savefig(path, bbox_inces="tight", dpi=300)
    plt.close()

    print(f"Saved: {path}")

def plot_roc(y_true, predictions_dict, family_name):
    """
    Saves the plot of ROC curve and computes AUC in a model family

    Parameters:
        y_true: array of true scores
        y_scores = predicted scores
        family_name: name of model family

    """
    plt.figure()
    plt.plot([0, 1], [1, 0], linestyle="--")


    for i, (model_name, y_scores) in enumerate(predictions_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC: {roc_auc:.3f})")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve -{family_name}")
    plt.legend()

    save_plot(family_name, "roc")

def plot_pr(y_true, predictions_dict, family_name):
    """Saves the plot of the Precision-Recall curve of a family model.
    
    Parameters:
        Parameters:
        y_true: array of true scores
        y_scores = predicted scores
        family_name: name of model family
        """
    plt.figure()

    for i, (model_name, y_scores) in enumerate(predictions_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, label=model_name)

    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve - {family_name}")

    save_plot(family_name, "pr")

def plot_sensitivity_fp(y_true, predictions_dict, family_name, n_points=50):
    plt.figure()

    thresholds = np.linspace(0, 1, n_points)

    for i, (model_name, y_scores) in enumerate(predictions_dict.items()):
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

        plt.plot(fps, sensitivities, label=model_name)

    plt.xlabel("False Positives (approx per sample)")
    plt.ylabel("Sensitivity")
    plt.title(f"FROC-style Curve - {family_name}")

    save_plot(family_name, "sens_fp")



def ensemble_predict(model_paths, X_test, val_aucs):
    """
    Combines predictions from multiple models using AUC-weighted sum.
    
    Parameters:
        model_paths: list of paths to best_model.keras files
        X_test: test features
        val_aucs: list of validation AUC scores, one per model (same order as model_paths)
    
    Returns:
        y_ensemble: weighted prediction array
        weights: the computed weights for reporting
    """
    # Compute normalised weights from validation AUCs
    total_auc = sum(val_aucs)
    weights = [auc / total_auc for auc in val_aucs]
    
    print("Ensemble weights:")
    for path, w, auc in zip(model_paths, weights, val_aucs):
        model_name = path.split(os.sep)[-2]
        print(f"  {model_name}: AUC={auc:.4f} → weight={w:.4f}")
    
    # Weighted sum of predictions
    y_ensemble = np.zeros(len(X_test))
    for path, weight in zip(model_paths, weights):
        model = keras.models.load_model(path)
        y_pred = model.predict(X_test).ravel()
        y_ensemble += weight * y_pred
    
    return y_ensemble, weights