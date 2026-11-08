import argparse
import matplotlib.pyplot as plt
import csv

from pathlib import Path


groups = {}

def get_plot_for_config(seed, learning_rate, num_epochs):
    runs = groups[(seed, learning_rate, num_epochs)]

    fig, ax = plt.subplots(figsize=(10, 5))

    for experiment, run in runs.items():
        epochs = run["epochs"]
        f1_scores = run["f1_scores"]

        ax.plot(epochs, f1_scores, label=experiment)

    ax.set_title(f"Validation F1 Scores - seed {seed}, learning_rate {learning_rate}, num_epochs {num_epochs}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("F1 Score")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.5)
    ax.set_xlim(1, num_epochs)

    fig.tight_layout()

    plots_path = Path("plots/") / f"{seed}_{learning_rate}_{num_epochs}-plot.png"
    plots_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(plots_path)
    plt.close(fig)

def fill_groups(preprocessing_experiments):
    for csv_path in sorted(Path("data").glob("*/*_learning-progress.csv")):
        # print(csv_path) # data/augmentation/augmentation_42_0.001_25_learning-progress.csv
        # print(csv_path.name) # augmentation_42_0.001_25_learning-progress.csv
        # print(csv_path.stem) # augmentation_42_0.001_25_learning-progress

        run_name = csv_path.stem.removesuffix("_learning-progress")
        experiment, seed, learning_rate, num_epochs = run_name.split("_")

        if experiment.startswith("preprocessing-"):
            preprocessing_name = experiment.removeprefix("preprocessing-")

            if preprocessing_name not in preprocessing_experiments:
                continue

        key = (int(seed), float(learning_rate), int(num_epochs))

        epochs=[]
        f1_scores=[]

        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                epochs.append(int(row["epoch"]))
                f1_scores.append(float(row["validation_f1"]))

        if key not in groups:
            groups[key] = {}

        groups[key][experiment] = {
            "epochs": epochs,
            "f1_scores": f1_scores
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessing",
        nargs="+",
        choices=["grayscale", "clahe", "grayscale-clahe"],
        default=["grayscale-clahe"]
    )
    args = parser.parse_args()

    fill_groups(args.preprocessing)
    for (seed, learning_rate, num_epochs) in groups:
        get_plot_for_config(seed, learning_rate, num_epochs)