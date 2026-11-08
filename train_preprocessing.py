import argparse

from train import setup_training, load_training_state, run_training, apply_grayscale, apply_clahe
from pathlib import Path

def main(run_parameters, num_epochs):
    (experiment, preprocessing) = run_parameters
    seed, train_dataloader, val_dataloader, model, loss_fn, optimizer, learning_rate = setup_training(preprocessing=preprocessing)

    experiment = f"preprocessing-{experiment}"
    run_name = f"{experiment}_{seed}_{learning_rate}_{num_epochs}"

    preprocessing_best_path = Path(f"model/{run_name}_best.pth")
    preprocessing_last_path = Path(f"model/{run_name}_last.pth")
    preprocessing_last_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = Path("data/preprocessing") / f"{run_name}_learning-progress.csv"

    start_epoch, best_f1 = load_training_state(preprocessing_last_path, model, optimizer)

    run_training(start_epoch, best_f1, seed, preprocessing_last_path, preprocessing_best_path, train_dataloader,
                 val_dataloader, model, loss_fn, optimizer, progress_path, num_epochs=num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument(
        "--experiment",
        choices=["grayscale", "clahe", "grayscale-clahe"],
        required=True)

    args = parser.parse_args()

    if args.num_epochs < 1:
        parser.error("--num_epochs must be 1 or greater")

    experiments = {
        "grayscale": ("grayscale", [apply_grayscale]),
        "clahe": ("clahe", [apply_clahe]),
        "grayscale-clahe": ("grayscale-clahe", [apply_grayscale, apply_clahe])
    }

    main(experiments[args.experiment], args.num_epochs)