# Unofficial Fork of DETR3D
This repo is an unofficial fork of [DETR3D](https://github.com/WangYueFt/detr3d).  
All credits belong to the original authors.  
For setup / replication of DETR3D experiments please refer to the original repository.

### Adaptions
Goal of this repository is to move DETR3D to the new mmdetection3d [rc1 coordinate conventions](https://mmdetection3d.readthedocs.io/en/latest/compatibility.html).
This results in the following adaptions:
- Moves bbox normalization / denormalization to new coordinate definition
- Packages detr3d to a pip-package

### Usage
- To use this fork: setup all dependencies of DETR3D as described in the original [README](https://github.com/WangYueFt/detr3d/blob/main/README.md).
- Install the DETR3D package by running `pip install -e .` in the repository root folder

