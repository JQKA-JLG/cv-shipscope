from train import setup_training, load_training_state, run_training
from pathlib import Path
from torchvision.transforms import v2

def main():
    random_rotation = v2.RandomChoice([
        v2.RandomRotation(degrees=(0, 0)),
        v2.RandomRotation(degrees=(90, 90)),
        v2.RandomRotation(degrees=(180, 180)),
        v2.RandomRotation(degrees=(270, 270))
    ])
    augmentation = [v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5), random_rotation]

    seed, train_dataloader, val_dataloader, model, loss_fn, optimizer, learning_rate = setup_training(augmentation=augmentation)

    experiment = "augmentation"
    run_name = f"{experiment}_{seed}_{learning_rate}"

    augmentation_best_path = Path(f"model/{run_name}_best.pth")
    augmentation_last_path = Path(f"model/{run_name}_last.pth")
    augmentation_last_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = Path(f"data/{experiment}/{run_name}_learning-progress.csv")

    start_epoch, best_f1 = load_training_state(augmentation_last_path, model, optimizer)

    run_training(start_epoch, best_f1, seed, augmentation_last_path, augmentation_best_path, train_dataloader,
                 val_dataloader, model, loss_fn, optimizer, progress_path)

if __name__ == "__main__":
    main()