#!/bin/bash

# this script should run inside the docker container
echo "installing mmdetection3d"
pip install -e /workspace/mmdetection3d

echo "installing DETR3d"
pip install -e /workspace/detr3d/

echo "installing SpatialDETR"
pip install -e /workspace

echo "DONE"