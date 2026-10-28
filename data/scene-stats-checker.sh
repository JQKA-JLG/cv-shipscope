#!/bin/bash

EXPECTED_CSV_LINES=435
EXPECTED_TOTAL_IMAGE_SUM=4000
EXPECTED_NEGATIVE_LABEL_SUM=3000
EXPECTED_POSITIVE_LABEL_SUM=1000

CSV="scene_stats.csv"

if [[ ! -f "$CSV" ]]; then
  printf 'There is no CSV file: %s\n' "$CSV" >&2
  exit 1
fi

actual_csv_lines=0
actual_total_image_sum=0
actual_negative_label_sum=0
actual_positive_label_sum=0

while IFS=',' read -r _ line_total line_negative line_positive; do
    actual_csv_lines=$((actual_csv_lines + 1))

    # CSV HEAD skip
    if (( actual_csv_lines == 1 )); then
        continue
    fi

    actual_total_image_sum=$((actual_total_image_sum + line_total))
    actual_negative_label_sum=$((actual_negative_label_sum + line_negative))
    actual_positive_label_sum=$((actual_positive_label_sum + line_positive))
done < "$CSV"

if [[ "$actual_csv_lines" != "$EXPECTED_CSV_LINES" ||
      "$actual_total_image_sum" != "$EXPECTED_TOTAL_IMAGE_SUM" ||
      "$actual_negative_label_sum" != "$EXPECTED_NEGATIVE_LABEL_SUM" ||
      "$actual_positive_label_sum" != "$EXPECTED_POSITIVE_LABEL_SUM" ]]; then

    printf 'CSV checks failed. Dataset does not match the expected values.\n' >&2
    printf 'expected csv_lines: %s; actual csv_lines: %s
expected total_image_sum: %s; actual total_image_sum: %s
expected negative_label_sum: %s; actual negative_label_sum: %s
expected positive_label_sum: %s; actual positive_label_sum: %s\n' \
    "$EXPECTED_CSV_LINES" "$actual_csv_lines" \
    "$EXPECTED_TOTAL_IMAGE_SUM" "$actual_total_image_sum" \
    "$EXPECTED_NEGATIVE_LABEL_SUM" "$actual_negative_label_sum" \
    "$EXPECTED_POSITIVE_LABEL_SUM" "$actual_positive_label_sum" >&2
    exit 1
fi

printf 'CSV checks passed.\n'

printf "Negative examples: %s\nPositive examples: %s\nTotal examples: %s\n" "$actual_negative_label_sum" "$actual_positive_label_sum" "$actual_total_image_sum"