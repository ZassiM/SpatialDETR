DOCKER_NAME=spatial_detr

# path to this repository root
repo_dir="C:\Users\wasso\Desktop\Project\SpatialDETR"
# path to directory where models / logs shall be stored in
exp_dir="C:\Users\wasso\Desktop\Project\SpatialDETR\work_dirs"

docker run \
--publish 6006 \
--gpus 'all,"capabilities=compute,utility,graphics"' \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$exp_dir,target=/workspace/work_dirs,type=bind,consistency=cached \
-it \
$DOCKER_NAME
