#!/usr/bin/env bash
export PYTHONPATH="."
python wonders_calculator/dataset_generation/generate_tfrecord.py \
    --images_dir=data/test_images/ \
    --csv_input=data/test_labels.csv  \
    --output_path=data/test.record