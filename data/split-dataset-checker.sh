#!/bin/bash

EXPECTED_TEST_FILES=600
EXPECTED_VALIDATE_FILES=600
EXPECTED_TRAIN_FILES=2800

cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

if (( $(find partitioned/train -maxdepth 1 -type f -printf '%f\n' | grep -Ec '^[01]__.*__.*\.png$') != EXPECTED_TRAIN_FILES  )); then
  printf "Train split image count does not match expectations.\n" >&2
  exit 1
fi

if (( $(find partitioned/test -maxdepth 1 -type f -printf '%f\n' | grep -Ec '^[01]__.*__.*\.png$') != EXPECTED_TEST_FILES  )); then
  printf "Test split image count does not match expectations.\n" >&2
  exit 1
fi

if (( $(find partitioned/validate -maxdepth 1 -type f -printf '%f\n' | grep -Ec '^[01]__.*__.*\.png$') != EXPECTED_VALIDATE_FILES  )); then
  printf "Validate split image count does not match expectations.\n" >&2
  exit 1
fi