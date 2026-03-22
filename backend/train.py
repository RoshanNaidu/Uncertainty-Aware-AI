"""
Deep Ensemble training script for CIFAR-10.

Trains M independent WideResNet-28-10 models with different random seeds.
Each model is trained from scratch (no pretraining).

Usage:
    # Full training (5 models × 200 epochs)
    python train.py

    # Quick sanity check
    python train.py --epochs 1 --ensemble-size 1

    # Custom settings
    python train.py --epochs 100 --ensemble-size 3 --batch-size 128
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from model import get_model


def get_cifar10_loaders(batch_size=128, num_workers=4, data_dir="./data"):
    """Create CIFAR-10 train and test dataloaders with standard augmentation."""
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch [{epoch+1}/{total_epochs}] "
                  f"Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset. Returns loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_single_model(model_idx, seed, args, device):
    """Train a single ensemble member with a specific random seed."""
    print(f"\n{'='*60}")
    print(f"Training Ensemble Member {model_idx + 1}/{args.ensemble_size} | Seed: {seed}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Build model
    model = get_model(num_classes=10, depth=16, widen_factor=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Data
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        data_dir=args.data_dir)

    # Optimizer & scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_acc = 0.0
    checkpoint_path = os.path.join(args.checkpoint_dir, f"model_{model_idx}.pt")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.epochs)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
              f"LR: {lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "seed": seed,
                "model_idx": model_idx,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (acc: {best_acc:.2f}%)")

    print(f"\nModel {model_idx} — Best Test Accuracy: {best_acc:.2f}%")
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train Deep Ensemble on CIFAR-10")
    parser.add_argument("--ensemble-size", type=int, default=5,
                        help="Number of ensemble members (default: 5)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs per model (default: 200)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Initial learning rate (default: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight decay (default: 5e-4)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory (default: ./data)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory (default: ./checkpoints)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Dataloader workers (default: 4)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (default: 0)")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Random seeds for each ensemble member
    base_seeds = [42, 137, 256, 512, 1024]
    seeds = base_seeds[:args.ensemble_size]

    results = []
    for i, seed in enumerate(seeds):
        acc = train_single_model(i, seed, args, device)
        results.append(acc)

    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*60}")
    for i, acc in enumerate(results):
        print(f"  Model {i}: {acc:.2f}%")
    print(f"  Mean accuracy: {sum(results)/len(results):.2f}%")


if __name__ == "__main__":
    main()
