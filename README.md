# ShipScope

**Scene-aware ship classification in satellite imagery**

ShipScope is a personal learning and portfolio project in computer vision.
Its goal is to classify satellite image patches as **ship** or **no ship**, with a focus on clear documentation, careful
evaluation and the analysis of prediction errors.

The key question is:
> How reliably can the model distinguish between ship and no-ship image patches from satellite scenes not used during 
training?

## Project status

This project is still in development. The first milestone is a working baseline evaluated on held-out scenes.
No measured model results are available yet.

## Dataset

This project uses **ShipsNet**, published by **rhammell** on Kaggle:
[Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery)

ShipsNet contains **4000 RGB images**, each measuring **80 x 80 pixels**. According to the dataset description, the images "were
derived from PlanetScope full-frame visual scene products, which are orthorectified to a 3-meter pixel size."

| Label | Class | Images |
|---|---|---:|
| 1 | Ship | 1000 |
| 0 | No ship | 3000 |

The patches come from PlanetScope imagery of the San Francisco Bay and San Pedro Bay areas in California. The no-ship 
class also includes partial ships and difficult negative examples.

The dataset is downloaded separately and excluded from version control.

## Planned Procedure

1. **Dataset audit & partitioning:** Export image counts per source scene and label to CSV, and check the overall totals 
    against the expected 4000 images: 3000 no-ship and 1000 ship examples. Then split the dataset into training, 
    validation, and test sets with target proportions of 70%, 15%, and 15%. Keep all images from the same source scene 
in a single split.
2. **Baseline training:** Train and evaluate a simple baseline.
3. **Experiment comparison:** Compare selected preprocessing methods, training augmentations and transfer learning. 
For each model, including the baseline, report precision, recall and F1-score for the ship class, present a confusion
matrix and analyse prediction errors.
4. **Command-line interface:** Add a command-line interface for new image patches.

## Planned stack
*Subject to change during development*

Python 3.12, PyTorch, torchvision, OpenCV, NumPy, Matplotlib and scikit-learn.

## Limitations

- The initial scope is classification of individual image patches.
- Different source scenes may cover the same geographic locations.
- Performance on ShipsNet alone does not establish reliability in other regions or with other satellite sensors.

## Licence and attribution

Original project code and documentation are licensed under the [MIT Licence](LICENSE).

ShipsNet and its source imagery are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The imagery was provided by **Planet** through its Open California programme. The dataset and source imagery are 
excluded from the project's MIT licence.