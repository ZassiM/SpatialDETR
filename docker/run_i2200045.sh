#!/bin/bash

DOCKER_NAME=xai_ma_zahr


# path to directory where nusenes data is stored
nusc_data_dir="/work/beemelmanns/nuscenes"
# nuscenes trainval
nusc_train_val_data_dir="/work/beemelmanns/nuscenes"
# path to directory where models / logs shall be stored in
work_dirs="/work/beemelmanns/work_dirs"
# path to this repository root
repo_dir="/work/beemelmanns/xai"

#--user $(id -u):$(id -g) \

docker run \
--name spatial_detr_container \
--gpus 'all,"capabilities=compute,utility,graphics"' \
--shm-size=16g \
--env DISPLAY=${DISPLAY} \
--net=host \
--volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
--volume $HOME/.Xauthority:/root/.Xauthority:rw \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$nusc_data_dir,target=/workspace/data/nuscenes_mini,type=bind,consistency=cached \
--mount source=$nusc_train_val_data_dir,target=/workspace/data/nuscenes,type=bind,consistency=cached \
--mount source=$work_dirs,target=/workspace/work_dirs,type=bind,consistency=cached \
-it \
$DOCKER_NAME
