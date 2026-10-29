import csv
import sys

CSV_PATH = "scene_stats.csv"
TARGET_NEG = 450
TARGET_POS = 150

# scene_id,split(train/validate/test)
OUTPUT_FILE = "scene_partitions.csv"

# Split the scenes in 3 Parts: train / validate / test
# Target is to find:
#       validate (15%): 450 negative, 150 positive
#       test (15%):     450 negative, 150 positive
#       train (70%):    the rest of scenes if validate and test were found (2100 negative, 700 positive)
# 2D-Subset-Sum for scenes that sums up to exactly (450, 150)


def load_scenes(path):
    scenes = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            scenes.append((row["scene_id"], int(row["negative"]), int(row["positive"])))
        return scenes

def sort_scenes(scenes):
    # Sort by sum of images descending, then by scene_id
    return sorted(scenes, key=lambda scene: (-(scene[1] + scene[2]), scene[0]))

# Where the magic happens
# Returns a List of scenes that sum is exactly (target_neg, target_pos)
def find_subset(scenes, target_neg, target_pos):
    target = (target_neg, target_pos)
    parent = {(0, 0): None}

    for i, (_, neg, pos) in enumerate(scenes):
        # Use a snapshot so the current scene cannot be selected more than once.
        for (n, p) in list(parent):
            next_state = (neg + n, pos + p)

            # do not add a child if it is not in the target range
            if next_state[0] > target_neg or next_state[1] > target_pos:
                continue

            # next_state is already accessible, remember first path
            if next_state in parent:
                continue

            parent[next_state] = (i, (n, p))

        # target got reached
        if target in parent:
            break

    if target not in parent:
        return None

    chosen = []
    state = target

    while parent[state] is not None:
        i, previous = parent[state]
        chosen.append(scenes[i])
        state = previous

    return chosen


def remove_scenes(scenes, scenes_to_remove):
    removed_scene_ids = {scene[0] for scene in scenes_to_remove}
    return [scene for scene in scenes if scene[0] not in removed_scene_ids]

def totals(scenes):
    sum_neg = sum(scene[1] for scene in scenes)
    sum_pos = sum(scene[2] for scene in scenes)
    return sum_neg, sum_pos

def save_scenes(splits):
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["scene_id", "split", "negative", "positive"])
        for split_id, scenes in splits.items():
            for scene_id, neg, pos in scenes:
                writer.writerow([scene_id, split_id, neg, pos])

def main():
    sorted_scenes = sort_scenes(load_scenes(CSV_PATH))

    validate = find_subset(sorted_scenes, TARGET_NEG, TARGET_POS)
    if validate is None:
        sys.exit("Did not find an combination for validate to reach target.")
    remaining_sorted_scenes = remove_scenes(sorted_scenes, validate)

    test = find_subset(remaining_sorted_scenes, TARGET_NEG, TARGET_POS)
    if test is None:
        sys.exit("Did not find an combination for test to reach target.")
    # train is what remains
    train = remove_scenes(remaining_sorted_scenes, test)

    assert totals(validate) == (TARGET_NEG, TARGET_POS)
    assert totals(test) == (TARGET_NEG, TARGET_POS)
    if totals(train) != (2100, 700):
        raise ValueError("Training split does not match the expected class totals.")

    save_scenes({"validate": validate, "test": test, "train": train})
    print("Done.\n")

if __name__ == "__main__":
    main()