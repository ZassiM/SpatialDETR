REPOMM3D=https://github.com/open-mmlab/mmdetection3d.git
MM3D=mmdetection3d
REPODETR3D=https://github.com/SimonDoll/detr3d.git
DETR3D=detr3d

LOCALREPO_MM3D=$MM3D/.git
LOCALREPO_DETR3D=$DETR3D/.git

DOCKER_NAME=spatial_detr

# path to directory where nusenes data is stored
nusc_data_dir="/work/data01/zahr/datasets/nuscenes"
# path to this repository root
repo_dir="/work/zahr/SpatialDETR"
# path to directory where models / logs shall be stored in
exp_dir="/work/zahr/SpatialDETR/work_dirs"

if [ ! -d $LOCALREPO_MM3D ]
then
    git clone --branch v1.0.0rc1 $REPOMM3D $MM3D
else
    cd $MM3D
    git pull --branch v1.0.0rc1 $REPOMM3D
    cd ..
fi

if [ ! -d $LOCALREPO_DETR3D ]
then
    git clone $REPODETR3D $DETR3D
else
    cd $DETR3D
    git pull $REPODETR3D
    cd ..
fi

docker build -t $DOCKER_NAME -f ./docker/Dockerfile .

docker run \
--rm \
--publish 6006 \
--gpus all --shm-size=16g \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$nusc_data_dir,target=/workspace/data/nuscenes,type=bind,consistency=cached \
--mount source=$exp_dir,target=/workspace/work_dirs,type=bind,consistency=cached \
-it \
$DOCKER_NAME

export nusc_data_dir
export repo_dir
export exp_dir
export DOCKER_NAME
