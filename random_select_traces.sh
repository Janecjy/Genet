#!/bin/bash

# Define source and destination directories
SOURCE_DIR="/users/janechen/Genet/fig_reproduce/data/synthetic_test_plus_mahimahi"
DEST_DIR="/users/janechen/Genet/fig_reproduce/data/synthetic_test_plus_mahimahi_subset"
ZIP_FILE="/users/janechen/Genet/fig_reproduce/data/synthetic_test_plus_mahimahi_subset.zip"
TRACE_NUM=$1

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find 10 random trace files and copy them to the subset directory
find "$SOURCE_DIR" -type f | shuf -n $TRACE_NUM | xargs -I {} cp {} "$DEST_DIR"

# Zip the subset directory
zip -r "$ZIP_FILE" "$DEST_DIR"

echo "Random selection completed and zipped to $ZIP_FILE"
