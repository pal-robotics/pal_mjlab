#!/bin/bash
set -e

usage() {
    echo "Usage: $0 --robot <robot_name> <source_dir>"
    exit 1
}

robot=""
src_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)
            robot="$2"
            shift 2
            ;;
        --robot=*)
            robot="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            src_dir="$1"
            shift
            ;;
    esac
done

if [[ -z "$robot" || -z "$src_dir" ]]; then
    usage
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
            --robot_name "$robot" \
            --output_name dummy

        mv /tmp/motion.npz "$out_dir/$name.npz"
    fi
done