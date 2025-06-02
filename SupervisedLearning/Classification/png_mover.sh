#!/bin/bash


SOURCE_DIR="/home/me/Documents/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees"
DEST_DIR="/mnt/data/MiscFiles/Datascience/SupervisedLearning/Classification/DecisionTrees/Plots"

mkdir -p "$DEST_DIR"

FILES_TO_MOVE=$(find "$SOURCE_DIR" -maxdepth 1 -type f -name "*.png")

if [ -z "$FILES_TO_MOVE" ]; then
    echo "No PNG files found in $SOURCE_DIR."
    exit 0  # Exit without error
fi

for file in "$SOURCE_DIR"/*.png; do
    [ -e "$file" ] || continue  # Skip if no files found
    mv "$file" "$DEST_DIR" && echo "Moved: $file"
done

# Logging moved files
echo "Moved $(ls "$DEST_DIR"/*.png | wc -l) PNG files to $DEST_DIR" >> move_files.log
echo "All PNG files have been successfully moved to $DEST_DIR."
