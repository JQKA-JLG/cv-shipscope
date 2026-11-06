import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

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

        # Backprogagation
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
            # Multiply the average loss by number of images ti get the batch's total loss.
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

def main():
    # Manual Seed for reproducibility
    # 42 - The Hitchhiker's Guide to the Galaxy reference
    seed=42
    torch.manual_seed(seed)

    train_dataset = ShipScopeDataset("data/partitioned/train")
    val_dataset = ShipScopeDataset("data/partitioned/validate")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = Net()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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