#!/usr/bin/env bash
export PYTHONPATH="."
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
protoc object_detection/protos/*.proto --python_out=.
jupyter notebook