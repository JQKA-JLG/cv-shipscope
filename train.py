import torch
import torch.nn as nn
import csv
from torch.utils.data import DataLoader
from pathlib import Path
from time import perf_counter

from dataset import ShipScopeDataset
from model import Net

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    loss_sum = 0.0
    processed = 0

    for batch, (features, labels) in enumerate(dataloader):
        # Compute prediction and loss
        prediction = model(features)
        loss = loss_fn(prediction, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_size = features.shape[0]
        batch_loss = loss.item()

        loss_sum += batch_loss * batch_size
        processed += batch_size

        if batch % 10 == 0 or processed == size:
            print(f"Batch loss: {batch_loss:.4f} [{processed}/{size}]")

    return loss_sum / processed

def validation_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    val_loss, correct = 0, 0

    # True Positive, True Negative, False Positive and False Negative counters for Confusion Matrix
    tp, tn, fp, fn = 0, 0, 0, 0

    with torch.no_grad():
        for features, labels in dataloader:
            prediction = model(features)
            predicted_labels = prediction.argmax(dim=1)
            # Multiply the average loss by the number of images to get the batch's total loss.
            val_loss += loss_fn(prediction, labels).item() * features.shape[0]
            correct += (predicted_labels == labels).type(torch.float).sum().item()
            tp += ((predicted_labels == 1) & (labels == 1)).sum().item()
            fp += ((predicted_labels == 1) & (labels == 0)).sum().item()
            fn += ((predicted_labels == 0) & (labels == 1)).sum().item()
            tn += ((predicted_labels == 0) & (labels == 0)).sum().item()

    # Validation Check
    assert tp + tn + fp + fn == size
    assert tp + tn == correct

    # '> 0 else 0.0' -> failsafe to ensure it won't get divided by zero
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    # Avg loss, Accuracy, Precision, Recall, F1
    return val_loss / size, correct / size, precision, recall, f1

def setup_training(seed=42, batch_size=64, learning_rate=0.001):
    torch.manual_seed(seed)
    train_dataset = ShipScopeDataset("data/partitioned/train")
    val_dataset = ShipScopeDataset("data/partitioned/validate")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Net()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return seed, train_dataloader, val_dataloader, model, loss_fn, optimizer, learning_rate

def load_training_state(last_path, model, optimizer):
    ts_start_epoch = 0
    ts_best_f1 = -1.0

    if last_path.exists():
        checkpoint = torch.load(last_path, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        ts_start_epoch = checkpoint["epoch"]
        ts_best_f1 = checkpoint["best_f1"]
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])

    return ts_start_epoch, ts_best_f1

def write_training_progress(progress_path, checkpoint, epoch_seconds):
    row = {
        "epoch": checkpoint["epoch"],
        "seed": checkpoint["seed"],
        "learning_rate": checkpoint["learning_rate"],
        "train_loss": checkpoint["train_loss"],
        "validation_loss": checkpoint["validation_loss"],
        "validation_accuracy": checkpoint["correct"],
        "validation_precision": checkpoint["precision"],
        "validation_recall": checkpoint["recall"],
        "validation_f1": checkpoint["f1"],
        "epoch_time_seconds": epoch_seconds
    }

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    has_content = progress_path.exists() and progress_path.stat().st_size > 0

    with progress_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))

        if not has_content:
            writer.writeheader()

        writer.writerow(row)

def run_training(start_epoch, best_f1, seed, last_path, best_path, train_dataloader, val_dataloader, model, loss_fn, optimizer, progress_path, num_epochs=25):
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        epoch_started = perf_counter()
        # Model training:
        epoch_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        print(f"Training loss: {epoch_loss:.4f}\n")

        # Model evaluation
        (epoch_val_loss,
         epoch_correct,
         epoch_precision,
         epoch_recall,
         epoch_f1) = validation_loop(val_dataloader, model, loss_fn)

        epoch_seconds = perf_counter() - epoch_started

        print(f"Validation: \n"
              f"Avg loss: {epoch_val_loss:.4f}\n"
              f"Accuracy: {epoch_correct:.1%}\n"
              f"Precision: {epoch_precision:.1%}\n"
              f"Recall: {epoch_recall:.1%}\n"
              f"F1: {epoch_f1:.1%}")
        is_best = epoch_f1 > best_f1
        best_f1 = max(best_f1, epoch_f1)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epoch_loss,
            "validation_loss": epoch_val_loss,
            "f1": epoch_f1,
            "best_f1": best_f1,
            "precision": epoch_precision,
            "recall": epoch_recall,
            "correct": epoch_correct,
            "seed": seed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "rng_state": torch.get_rng_state()
        }
        # Save latest state after every completed epoch.
        torch.save(checkpoint, last_path)

        # Write progress
        write_training_progress(progress_path, checkpoint, epoch_seconds)

        # Replace the best checkpoint when F1 improves in current epoch.
        if is_best:
            torch.save(checkpoint, best_path)
            print(f"New Best F1: {best_f1:.1%}")
    print(f"Best F1: {best_f1:.1%}")

def main():
    # Manual Seed for reproducibility
    # 42 - The Hitchhiker's Guide to the Galaxy reference
    seed, train_dataloader, val_dataloader, model, loss_fn, optimizer, _ = setup_training()

    checkpoint_path = Path("model/cnn_debug_best.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
    except FileNotFoundError:
        checkpoint = None

    best_f1 = checkpoint["f1"] if checkpoint is not None else -1.0

    for epoch in range(20):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        epoch_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        print(f"Training loss: {epoch_loss:.4f}\n")

        (epoch_val_loss,
         epoch_correct,
         epoch_precision,
         epoch_recall,
         epoch_f1) = validation_loop(val_dataloader, model, loss_fn)
        print(f"Validation: \n"
              f"Avg loss: {epoch_val_loss:.4f}\n"
              f"Accuracy: {epoch_correct:.1%}, "
              f"Precision: {epoch_precision:.1%}, "
              f"Recall: {epoch_recall:.1%}, "
              f"F1: {epoch_f1:.1%}")
        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_loss,
                "validation_loss": epoch_val_loss,
                "f1": epoch_f1,
                "precision": epoch_precision,
                "recall": epoch_recall,
                "correct": epoch_correct,
                "seed": seed,
                "learning_rate": optimizer.param_groups[0]["lr"]
            }, checkpoint_path)
            print(f"New Best F1: {best_f1:.1%}")

    print(f"Best F1: {best_f1:.1%}")

if __name__ == "__main__":
    main()