DOCKER_NAME=spatial_detr

# path to directory where nusenes data is stored
nusc_data_dir="/home/zahr/Documents/datasets/nuscenes"
# path to this repository root
repo_dir="/home/zahr/Documents/SpatialDETR"
# path to directory where models / logs shall be stored in
exp_dir="/home/zahr/Documents/SpatialDETR/work_dirs"

XSOCK=/tmp/.X11-unix  # to support X11 forwarding in isolated containers on local host (not thorugh SSH)
# XAUTH=$(mktemp /tmp/.docker.xauth.XXXXXXXXX)
# xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge - 2>/dev/null
# chmod 777 $XAUTH

docker run \
--rm \
--publish 6006 \
--gpus 'all,"capabilities=compute,utility,graphics"' \
--env QT_X11_NO_MITSHM=1 \
--env DISPLAY=${DISPLAY} \
--env XAUTHORITY=${XAUTH} \
--volume ${XAUTH}:${XAUTH} \
--volume ${XSOCK}:${XSOCK} \
--shm-size=16g \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$nusc_data_dir,target=/workspace/data/nuscenes,type=bind,consistency=cached \
--mount source=$exp_dir,target=/workspace/work_dirs,type=bind,consistency=cached \
-it \
$DOCKER_NAME
