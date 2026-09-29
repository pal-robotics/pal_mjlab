#!/bin/bash

MOTION_DIR="$1"
ROBOT_NAME="${2:-kangaroo}"

if [ -z "$MOTION_DIR" ]; then
    echo "Usage: $0 <motion_dir> [robot_name]"
    exit 1
fi

if [ ! -d "$MOTION_DIR" ]; then
    echo "Error: directory '$MOTION_DIR' does not exist."
    exit 1
fi

for csv_file in "$MOTION_DIR"/*.csv; do
    [ -e "$csv_file" ] || continue

    filename=$(basename "$csv_file" .csv)
    output_file="$MOTION_DIR/$filename.npz"

    echo "========================================"
    echo "Processing: $csv_file"
    echo "Output:    $output_file"
    echo "========================================"

    uv run -m pal_mjlab.scripts.csv_to_npz \
        --input-file "$csv_file" \
        --output-name "$filename" \
        --input-fps 30 \
        --output-fps 50 \
        --render False \
        --robot_name "$ROBOT_NAME"

    if [ $? -ne 0 ]; then
        echo "ERROR: conversion failed for $csv_file"
        continue
    fi

    if [ ! -f /tmp/motion.npz ]; then
        echo "ERROR: /tmp/motion.npz was not created"
        continue
    fi

    mv /tmp/motion.npz "$output_file"

    echo "Saved: $output_file"
done
