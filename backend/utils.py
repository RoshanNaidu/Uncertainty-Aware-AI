"""
Uncertainty quantification utilities for Deep Ensembles.

Provides:
- Ensemble prediction aggregation
- Predictive Entropy (total uncertainty)
- Mutual Information (epistemic uncertainty)
- Expected Calibration Error (ECE)
- AUROC for OOD detection
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def ensemble_predict(models, x, device="cpu"):
    """
    Run inference through all ensemble members and return per-model softmax
    probabilities.

    Parameters
    ----------
    models : list[nn.Module]
        List of M ensemble members.
    x : torch.Tensor
        Input batch of shape (B, C, H, W).
    device : str or torch.device

    Returns
    -------
    probs : np.ndarray of shape (M, B, K)
        Softmax probabilities from each model.
    """
    all_probs = []
    x = x.to(device)
    for model in models:
        model.eval()
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.stack(all_probs, axis=0)  # (M, B, K)


def mean_prediction(probs):
    """Mean predictive probability across ensemble members.

    Parameters
    ----------
    probs : np.ndarray of shape (M, B, K)

    Returns
    -------
    mean_probs : np.ndarray of shape (B, K)
    """
    return probs.mean(axis=0)


# ---------------------------------------------------------------------------
# Uncertainty metrics
# ---------------------------------------------------------------------------

def predictive_entropy(mean_probs):
    """
    H[E[p]] — total uncertainty (predictive entropy).

    Parameters
    ----------
    mean_probs : np.ndarray of shape (B, K)
        Mean predictive probabilities.

    Returns
    -------
    entropy : np.ndarray of shape (B,)
    """
    eps = 1e-10
    return -np.sum(mean_probs * np.log(mean_probs + eps), axis=-1)


def mutual_information(probs):
    """
    MI = H[E[p]] - E[H[p]] — epistemic uncertainty (model disagreement).

    Parameters
    ----------
    probs : np.ndarray of shape (M, B, K)

    Returns
    -------
    mi : np.ndarray of shape (B,)
    """
    eps = 1e-10
    mean_p = mean_prediction(probs)
    total_entropy = predictive_entropy(mean_p)

    # Expected entropy across ensemble members
    member_entropies = -np.sum(probs * np.log(probs + eps), axis=-1)  # (M, B)
    expected_entropy = member_entropies.mean(axis=0)  # (B,)

    return total_entropy - expected_entropy


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(confidences, accuracies, num_bins=15):
    """
    Expected Calibration Error (ECE).

    Parameters
    ----------
    confidences : np.ndarray of shape (N,)
        Predicted confidence (max probability) for each sample.
    accuracies : np.ndarray of shape (N,)
        Binary array: 1 if prediction is correct, 0 otherwise.
    num_bins : int

    Returns
    -------
    ece : float
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n_total = len(confidences)

    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (n_bin / n_total) * abs(avg_acc - avg_conf)

    return ece


# ---------------------------------------------------------------------------
# OOD detection
# ---------------------------------------------------------------------------

def ood_auroc(id_scores, ood_scores):
    """
    AUROC for OOD detection.

    Higher uncertainty scores for OOD samples → higher AUROC.

    Parameters
    ----------
    id_scores : np.ndarray
        Uncertainty scores for in-distribution samples.
    ood_scores : np.ndarray
        Uncertainty scores for out-of-distribution samples.

    Returns
    -------
    auroc : float
    """
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(labels, scores)


# ---------------------------------------------------------------------------
# CIFAR-10 class names
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
