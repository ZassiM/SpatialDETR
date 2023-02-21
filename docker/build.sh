REPOMM3D=https://github.com/open-mmlab/mmdetection3d.git
MM3D=mmdetection3d
REPODETR3D=https://github.com/SimonDoll/detr3d.git
DETR3D=detr3d

LOCALREPO_MM3D=$MM3D/.git
LOCALREPO_DETR3D=$DETR3D/.git

DOCKER_NAME=spatial_detr

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