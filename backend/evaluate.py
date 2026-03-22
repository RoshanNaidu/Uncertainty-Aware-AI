"""
Evaluation script for the Deep Ensemble.

Computes:
- Accuracy on CIFAR-10 test set
- Expected Calibration Error (ECE)
- OOD detection AUROC (CIFAR-10 vs CIFAR-100 using predictive entropy)

Usage:
    python evaluate.py
    python evaluate.py --checkpoint-dir ./checkpoints --ensemble-size 5
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from model import get_model
from utils import (
    ensemble_predict, mean_prediction, predictive_entropy,
    mutual_information, expected_calibration_error, ood_auroc,
    CIFAR10_CLASSES,
)


def load_ensemble(checkpoint_dir, ensemble_size, device, num_classes=10):
    """Load all ensemble member models from checkpoints."""
    models = []
    for i in range(ensemble_size):
        path = os.path.join(checkpoint_dir, f"model_{i}.pt")
        if not os.path.exists(path):
            print(f"Warning: Checkpoint not found: {path}")
            continue

        model = get_model(num_classes=num_classes, depth=16, widen_factor=2)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
        print(f"  Loaded model_{i} (test_acc: {checkpoint.get('test_acc', '?'):.2f}%, "
              f"seed: {checkpoint.get('seed', '?')})")

    print(f"Loaded {len(models)}/{ensemble_size} ensemble members")
    return models


def get_test_loader(dataset_class, data_dir="./data", batch_size=128, num_workers=4):
    """Create a test dataloader with standard normalization."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2470, 0.2435, 0.2616)),
    ])
    dataset = dataset_class(
        root=data_dir, train=False, download=True, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)


def evaluate_accuracy_and_calibration(models, loader, device):
    """Compute ensemble accuracy and ECE on a dataset."""
    all_probs = []
    all_targets = []

    for inputs, targets in loader:
        probs = ensemble_predict(models, inputs, device)  # (M, B, K)
        all_probs.append(probs)
        all_targets.append(targets.numpy())

    all_probs = np.concatenate(all_probs, axis=1)    # (M, N, K)
    all_targets = np.concatenate(all_targets, axis=0)  # (N,)

    mean_p = mean_prediction(all_probs)                # (N, K)
    predictions = np.argmax(mean_p, axis=1)            # (N,)
    confidences = np.max(mean_p, axis=1)               # (N,)
    accuracies_bin = (predictions == all_targets).astype(float)

    accuracy = 100.0 * accuracies_bin.mean()
    ece = expected_calibration_error(confidences, accuracies_bin)

    return accuracy, ece, all_probs


def compute_ood_scores(models, id_loader, ood_loader, device):
    """Compute uncertainty scores for ID and OOD data."""
    id_probs_list = []
    ood_probs_list = []

    print("Computing ID (CIFAR-10) uncertainty scores...")
    for inputs, _ in id_loader:
        probs = ensemble_predict(models, inputs, device)
        id_probs_list.append(probs)

    print("Computing OOD (CIFAR-100) uncertainty scores...")
    for inputs, _ in ood_loader:
        probs = ensemble_predict(models, inputs, device)
        ood_probs_list.append(probs)

    id_probs = np.concatenate(id_probs_list, axis=1)    # (M, N_id, K)
    ood_probs = np.concatenate(ood_probs_list, axis=1)  # (M, N_ood, K)

    # Predictive entropy as OOD score
    id_entropy = predictive_entropy(mean_prediction(id_probs))
    ood_entropy = predictive_entropy(mean_prediction(ood_probs))

    # Mutual information as OOD score
    id_mi = mutual_information(id_probs)
    ood_mi = mutual_information(ood_probs)

    return {
        "id_entropy": id_entropy,
        "ood_entropy": ood_entropy,
        "id_mi": id_mi,
        "ood_mi": ood_mi,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Deep Ensemble")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load ensemble
    print("\nLoading ensemble...")
    models = load_ensemble(args.checkpoint_dir, args.ensemble_size, device)
    if not models:
        print("ERROR: No models loaded. Train models first with train.py")
        return

    # Evaluate on CIFAR-10 (in-distribution)
    print("\n" + "=" * 60)
    print("IN-DISTRIBUTION EVALUATION (CIFAR-10)")
    print("=" * 60)
    cifar10_loader = get_test_loader(
        torchvision.datasets.CIFAR10, args.data_dir,
        args.batch_size, args.num_workers)
    accuracy, ece, _ = evaluate_accuracy_and_calibration(
        models, cifar10_loader, device)
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  ECE:      {ece:.4f}")

    # OOD detection (CIFAR-10 vs CIFAR-100)
    print("\n" + "=" * 60)
    print("OOD DETECTION (CIFAR-10 vs CIFAR-100)")
    print("=" * 60)
    cifar100_loader = get_test_loader(
        torchvision.datasets.CIFAR100, args.data_dir,
        args.batch_size, args.num_workers)
    scores = compute_ood_scores(models, cifar10_loader, cifar100_loader, device)

    auroc_entropy = ood_auroc(scores["id_entropy"], scores["ood_entropy"])
    auroc_mi = ood_auroc(scores["id_mi"], scores["ood_mi"])
    print(f"  AUROC (Predictive Entropy): {auroc_entropy:.4f}")
    print(f"  AUROC (Mutual Information): {auroc_mi:.4f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("UNCERTAINTY STATISTICS")
    print("=" * 60)
    print(f"  ID Entropy  — mean: {scores['id_entropy'].mean():.4f}, "
          f"std: {scores['id_entropy'].std():.4f}")
    print(f"  OOD Entropy — mean: {scores['ood_entropy'].mean():.4f}, "
          f"std: {scores['ood_entropy'].std():.4f}")
    print(f"  ID MI       — mean: {scores['id_mi'].mean():.4f}, "
          f"std: {scores['id_mi'].std():.4f}")
    print(f"  OOD MI      — mean: {scores['ood_mi'].mean():.4f}, "
          f"std: {scores['ood_mi'].std():.4f}")

    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
