#!/usr/bin/env bash
export PYTHONPATH="."
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
protoc object_detection/protos/*.proto --python_out=.
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path=train/models/object_detection_pipeline_pet.config \
    --trained_checkpoint_prefix train/models/train/model.ckpt-137604 \
    --output_directory train/models/output_model
