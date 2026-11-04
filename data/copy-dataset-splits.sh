#!/bin/bash

DATASET_PATH='archive/shipsnet/shipsnet/'

SCENE_PARTITIONS_CSV='scene_partitions.csv'

PARTITIONS_PATH='partitioned/'

cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

if ! ./dataset-audit.sh; then
  exit 1
fi

mkdir -p -- \
  "${PARTITIONS_PATH}train" \
  "${PARTITIONS_PATH}validate" \
  "${PARTITIONS_PATH}test" || exit 1

if ! python3 "dataset-partioner.py"; then
  printf "Copy Dataset splits failed. dataset-partitoner.py returned error exitcode.\n" >&2
  exit 1
fi

if [[ ! -f "$SCENE_PARTITIONS_CSV" ]]; then
  printf "scene_partitions.csv not found.\n" >&2
  exit 1
fi

while IFS=',' read -r line_scene_id line_split _ _; do
  while read -r image_name; do
    if ! cp -- "$DATASET_PATH$image_name" "$PARTITIONS_PATH$line_split/$image_name"; then
      printf "Failed to copy image: %s\n" "$image_name" >&2
      exit 1
    fi
  done < <(find "$DATASET_PATH" -maxdepth 1 -type f -printf '%f\n' | grep -E "^[01]__${line_scene_id}__.*\.png$")
done < "$SCENE_PARTITIONS_CSV"

if ! ./split-dataset-checker.sh; then
  printf "Split dataset check was not successful.\n" >&2
  exit 1
fi

printf "Dataset was split successfully.\n"