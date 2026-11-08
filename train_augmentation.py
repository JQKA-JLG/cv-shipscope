import argparse

from train import setup_training, load_training_state, run_training
from pathlib import Path
from torchvision.transforms import v2

def main(num_epochs, learning_rate, seed):
    random_rotation = v2.RandomChoice([
        v2.RandomRotation(degrees=(0, 0)),
        v2.RandomRotation(degrees=(90, 90)),
        v2.RandomRotation(degrees=(180, 180)),
        v2.RandomRotation(degrees=(270, 270))
    ])
    augmentation = [v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5), random_rotation]

    seed, train_dataloader, val_dataloader, model, loss_fn, optimizer, learning_rate = setup_training(augmentation=augmentation, learning_rate=learning_rate, seed=seed)

    experiment = "augmentation"
    run_name = f"{experiment}_{seed}_{learning_rate}_{num_epochs}"

    augmentation_best_path = Path(f"model/{run_name}_best.pth")
    augmentation_last_path = Path(f"model/{run_name}_last.pth")
    augmentation_last_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = Path(f"data/{experiment}/{run_name}_learning-progress.csv")

    start_epoch, best_f1 = load_training_state(augmentation_last_path, model, optimizer)

    run_training(start_epoch, best_f1, seed, augmentation_last_path, augmentation_best_path, train_dataloader,
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

    main(args.num_epochs, args.learning_rate, args.seed)