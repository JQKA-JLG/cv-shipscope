import argparse

from train import setup_training, load_training_state, run_training
from pathlib import Path

def main(num_epochs, learning_rate, seed):
    seed, train_dataloader, val_dataloader, model, loss_fn, optimizer, learning_rate = setup_training(learning_rate=learning_rate, seed=seed)

    experiment = "baseline"
    run_name = f"{experiment}_{seed}_{learning_rate}_{num_epochs}"

    baseline_best_path = Path(f"model/{run_name}_best.pth")
    baseline_last_path = Path(f"model/{run_name}_last.pth")
    baseline_last_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = Path(f"data/{experiment}/{run_name}_learning-progress.csv")

    start_epoch, best_f1 = load_training_state(baseline_last_path, model, optimizer)

    run_training(start_epoch, best_f1, seed, baseline_last_path, baseline_best_path, train_dataloader,
                 val_dataloader, model, loss_fn, optimizer, progress_path, num_epochs=num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_epochs < 1:
        parser.error("--num_epochs must be 1 or greater")

    if args.learning_rate <= 0:
        parser.error("--learning_rate must be greater than 0")

    main(num_epochs=args.num_epochs, learning_rate=args.learning_rate, seed=args.seed)