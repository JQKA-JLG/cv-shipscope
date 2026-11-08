import torch
import torch.nn as nn
import csv

from pathlib import Path
from torch.utils.data import DataLoader
from model import Net
from dataset import ShipScopeDataset
from train import validation_loop, apply_grayscale, apply_clahe
from cv_plotter import plot_confusion_matrix

def main(checkpoint_best_path):
    preprocessing_by_experiment = {
        "baseline": None,
        "augmentation": None,
        "preprocessing-grayscale": [apply_grayscale],
        "preprocessing-clahe": [apply_clahe],
        "preprocessing-grayscale-clahe": [apply_grayscale, apply_clahe]
    }

    run_name = checkpoint_best_path.stem.removesuffix("_best")
    experiment, seed, learning_rate, num_epochs = run_name.split("_", 3)

    test_dataset = ShipScopeDataset(
        "data/partitioned/test",
        preprocessing=preprocessing_by_experiment[experiment]
    )
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Net()
    loss_fn = nn.CrossEntropyLoss()

    load_state(checkpoint_best_path, model)

    metrics, confusion_matrix = validation_loop(test_dataloader, model, loss_fn, return_confusion_matrix=True)

    print(f"{checkpoint_best_path.name}: F1 = {metrics[4]:.2%}")

    save_evaluation(checkpoint_best_path, metrics, confusion_matrix)

def load_state(best_path, model):
    if best_path.exists():
        checkpoint = torch.load(best_path, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise FileNotFoundError

def save_evaluation(checkpoint_path, metrics, confusion_matrix):
    run_name = Path(checkpoint_path).stem.removesuffix("_best")
    experiment, seed, learning_rate, num_epochs = run_name.split("_", 3)

    loss, accuracy, precision, recall, f1 = metrics
    (tn, fp), (fn, tp) = confusion_matrix

    result = {
        "checkpoint": checkpoint_path.name,
        "experiment": experiment,
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "num_epochs": int(num_epochs),
        "loss": float(loss),
        "split": "test",
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    output_path = Path("data/evaluation") / f"{checkpoint_path.stem}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

if __name__ == "__main__":
    for checkpoint_path in sorted(Path("model").glob("*_best.pth")):
        main(checkpoint_path)

    for evaluation_path in sorted(Path("data/evaluation").glob("*.csv")):
        with open(evaluation_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            test_evaluation = next(reader)
        plot_confusion_matrix(test_evaluation)