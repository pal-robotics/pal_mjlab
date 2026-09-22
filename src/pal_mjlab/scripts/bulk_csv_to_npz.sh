#!/bin/bash
set -e

# PASS SOURCE DIRECTORY AS FIRST ARGUMENT
src_dir="$1"

if [[ -z "$src_dir" ]]; then
    echo "Usage: $0 <source_dir>"
    exit 1
fi

out_dir="$src_dir"

for path in "$src_dir"/*; do
    file=$(basename "$path")

    if [[ "$file" == *.csv ]]; then
        name="${file%.csv}"

        uv run -m pal_mjlab.scripts.csv_to_npz \
            --input-file "$path" \
            --input-fps 30 \
            --output-fps 50 \
            --render False \
            --robot_name kangaroo \
            --output_name dummy

        mv /tmp/motion.npz "$out_dir/$name.npz"
    fi
done
