from train import setup_training, load_training_state, run_training
from pathlib import Path

def main():
    seed, train_dataloader, val_dataloader, model, loss_fn, optimizer, learning_rate = setup_training()

    experiment = "baseline"

    run_name = f"{experiment}_{seed}_{learning_rate}"

    baseline_best_path = Path(f"model/{run_name}_best.pth")
    baseline_last_path = Path(f"model/{run_name}_last.pth")
    baseline_last_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = Path(f"data/{experiment}/{run_name}_learning-progress.csv")

    start_epoch, best_f1 = load_training_state(baseline_last_path, model, optimizer)

    run_training(start_epoch, best_f1, seed, baseline_last_path, baseline_best_path, train_dataloader,
                 val_dataloader, model, loss_fn, optimizer, progress_path)

if __name__ == "__main__":
    main()