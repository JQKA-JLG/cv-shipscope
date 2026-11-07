from train import setup_training, load_training_state, run_training
from pathlib import Path

def main():
    seed, train_dataloader, val_dataloader, model, loss_fn, optimizer = setup_training()

    baseline_best_path = Path("model/baseline_model_best.pth")
    baseline_last_path = Path("model/baseline_model_last.pth")
    baseline_last_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch, best_f1 = load_training_state(baseline_last_path, model, optimizer)

    run_training(start_epoch, best_f1, seed, baseline_last_path, baseline_best_path, train_dataloader,
                 val_dataloader, model, loss_fn, optimizer)

if __name__ == "__main__":
    main()