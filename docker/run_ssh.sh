#!/bin/bash

DOCKER_NAME=spatial_detr

# path to directory where nusenes data is stored
nusc_data_dir="/work/data01/zahr/datasets/nuscenes"
nusc_train_val_data_dir="/work/data01/beemelmanns/nuscenes"
# path to this repository root
repo_dir=$PWD
# path to directory where models / logs shall be stored in
work_dirs="/work/zahr/SpatialDETR/work_dirs"

XSOCK=/tmp/.X11-unix

docker run \
--rm \
--gpus 'all,"capabilities=compute,utility,graphics"' \
-e DISPLAY \
--net=host \
--volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
--volume $HOME/.Xauthority:/root/.Xauthority:rw \
--shm-size=16g \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$nusc_data_dir,target=/workspace/data/nuscenes,type=bind,consistency=cached \
--mount source=$nusc_train_val_data_dir,target=/workspace/data/nuscenes_trainval,type=bind,consistency=cached \
--mount source=$work_dirs,target=/workspace/work_dirs,type=bind,consistency=cached \
-it \
$DOCKER_NAME