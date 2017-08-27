#!/usr/bin/env bash
export PYTHONPATH="."
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
protoc object_detection/protos/*.proto --python_out=.
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=train/models/object_detection_pipeline_pet.config \
    --train_dir=train/models/train/