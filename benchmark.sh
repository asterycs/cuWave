#!/bin/bash

SCENEFILE="$1"
RUNS="$2"
if [ -z "$RUNS" ]; then
  PATHS=1
fi
OUT=$(./cuWave -b -s "$SCENEFILE" -o out.png -p "$RUNS")

TIME=$(echo "$OUT" | grep "Rendering time" | awk '{print $NF}')

echo "$TIME" >> benchmarks.txt

echo "$TIME" milliseconds

exit 0
