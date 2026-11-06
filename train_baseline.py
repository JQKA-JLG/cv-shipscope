import torch

from train import train_loop, validation_loop, setup_training
from pathlib import Path

def main():
    seed, train_dataloader, val_dataloader, model, loss_fn, optimizer = setup_training()

    baseline_best_path = Path("model/baseline_model_best.pth")
    baseline_last_path = Path("model/baseline_model_last.pth")
    baseline_last_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_f1 = -1.0

    if baseline_last_path.exists():
        checkpoint = torch.load(baseline_last_path, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_f1 = checkpoint["best_f1"]
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])

    for epoch in range(start_epoch, 25):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        # Baseline Model training:
        epoch_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        print(f"Training loss: {epoch_loss:.4f}\n")

        # Baseline Model evaluation
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
        torch.save(checkpoint, baseline_last_path)

        # Replace the best checkpoint when F1 improves in current epoch.
        if is_best:
            torch.save(checkpoint, baseline_best_path)
            print(f"New Best F1: {best_f1:.1%}")
    print(f"Best F1: {best_f1:.1%}")

if __name__ == "__main__":
    main()