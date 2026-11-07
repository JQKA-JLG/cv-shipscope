from train import setup_training, load_training_state, run_training, apply_grayscale, apply_clahe
from pathlib import Path

def main(run_parameters):
    (experiment, preprocessing) = run_parameters
    seed, train_dataloader, val_dataloader, model, loss_fn, optimizer, learning_rate = setup_training(preprocessing=preprocessing)

    run_name = f"{experiment}_{seed}_{learning_rate}"

    preprocessing_best_path = Path(f"model/{run_name}_best.pth")
    preprocessing_last_path = Path(f"model/{run_name}_last.pth")
    preprocessing_last_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = Path(f"data/{experiment}/{run_name}_learning-progress.csv")

    start_epoch, best_f1 = load_training_state(preprocessing_last_path, model, optimizer)

    run_training(start_epoch, best_f1, seed, preprocessing_last_path, preprocessing_best_path, train_dataloader,
                 val_dataloader, model, loss_fn, optimizer, progress_path)

if __name__ == "__main__":
    run_parameters_tuples = [
        ("grayscale", [apply_grayscale]),
        ("clahe", [apply_clahe]),
        ("grayscale_clahe", [apply_grayscale, apply_clahe])]

    for run_parameters in run_parameters_tuples:
        main(run_parameters)