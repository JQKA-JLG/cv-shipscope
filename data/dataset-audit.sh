#!/bin/bash

DATASET_PATH='archive/shipsnet/shipsnet/'

if [[ ! -d "$DATASET_PATH" ]]; then
  printf 'Dataset directory not found: %s\n' "$DATASET_PATH" >&2
  exit 1
fi

printf "Scene infos:\n"
find "$DATASET_PATH" -maxdepth 1 -type f -printf '%f\n' |
  sed -nE 's/^[01]__(.*)__.*\.png$/\1/p' |
  sort |
  uniq -c |
  awk 'END { printf "Number of unique scenes: %d\n", NR }'

printf "\nScene-stats as CSV:\n"
find "$DATASET_PATH" -maxdepth 1 -type f -printf '%f\n' |   sed -nE 's/^([01])__(.*)__.*\.png$/\2 \1/p' |
  awk 'BEGIN {
    print "scene_id,total_scene_images,negative,positive"
    }
    {
      total[$1]++

      if ($2 == 0) {
        negative[$1]++
      } else {
        positive[$1]++
      }
  }
  END {
    for (scene in total) {
      printf "%s,%d,%d,%d\n", scene, total[scene], negative[scene], positive[scene]
    }
  }' |
    {
      IFS= read -r csv_header;
      printf '%s\n' "$csv_header";
      sort;
    }  |
      tee scene_stats.csv

if  ! ./scene-stats-checker.sh; then
  printf "Dataset check failed. See error output above.\n" >&2
  printf "Gandalf: You shall not pass!\n" >&2
  exit 1
fi

printf "\nDataset statistics match the expected values.\n"