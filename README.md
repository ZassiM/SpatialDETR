# Explainable Transformer-based 3D Object Detector from multiple cameras


## Setup
### Repository
1. Clone the repository together with its submodules: 

```bash
git clone --recurse-submodules git@gitlab.ika.rwth-aachen.de:ma-zahr/spatialdetr.git SpatialDETR
```

2. If you have already cloned the repository without the --recurse-submodules flag, initialize all submodules manually: 

```bash
git submodule update --init --recursive
```

### Nuscenes Dataset
1. Follow the [mmdetection3d instructions](https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/datasets/nuscenes_det.html) to preprocess the data of the nuScenes dataset.

### Model weights
1. Download the [weights for FCOS3D](https://rwth-aachen.sciebo.de/s/asoSC5oMD1TNEsy) (which are used by SpatialDETR) and put them inside a directory called **pretrained**.
2. Download the [weights for SpatialDETR](https://rwth-aachen.sciebo.de/s/fgmMdPEQKQu9hz) (query_proj_value_proj) and put them inside a directory called **checkpoints**.


### Docker Setup
The docker container is based on the following packages:
- `Python 3.7`
- `Pytorch 1.9`
- `Cuda 11.1`
- `MMCV 1.5`
- `MMDetection 2.23`
- `MMSegmentation 0.20`

To ease the setup of this project we provide a docker container and some convenience scripts (see `docker`). To setup use:
- (if not done alread) setup [Docker](https://docs.docker.com/engine/install/ubuntu/) and [Nvidia-Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- Run 
```bash
./docker/build.sh
```
to build the container
- If on a Linux machine, open `bash /docker/run_loc.sh`. If on a Windows machine `bash /docker/run_win.sh`. If on a remote Linux server, open `/docker/run_cluster.sh`.
- Adapt the `nusc_data_dir` to the nuscenes directory and `work_dirs` to the direcory where model weights and pretrained weights directories are saved.

### Container Run and Setup
1. If on Linux machine run:
```bash
./docker/run_loc.sh
```
If on Windows machine run:
```bash
./docker/run_win.sh
```
Otherwise, if working on a remote server, run:
```bash
./docker/run_cluster.sh
```
2. Now the container is running, but some packages still need to be installed. Run 
```bash
./docker/in_docker_setup.sh
```

## Application testing
1. After the docker container is set-up, run the application:
```bash 
python main.py
```
2. A GUI will appear. Select **File-Load model** to load the model configuration and the checkpoints. Otherwise, if the same configurations are used each time, modify the **config.toml** file accordinly and select **File-Load from config file**.
3. Change the visualization settings with the drop-down menus. Then, click **Visualize**. 

## Model Training
1. Use the configs in the **configs** folder to train SpatialDETR.  
For a baseline on a single gpu use:
```bash
python ./mmdetection3d/tools/train.py configs/submission/frozen_4/query_proj_value_proj.py
```
or for multi-gpu e.g. 4 gpus:  
```bash
./mmdetection3d/tools/dist_train.sh configs/submission/frozen_4/query_proj_value_proj.py 4
```

3. To test, use
```bash
./mmdetection3d/tools/dist_test.sh configs/submission/frozen_4/query_proj_value_proj.py /path/to/.pth 4 --eval=bbox
```

## Issues and To-Do
- The shell for running the docker container makes sure that the display environment is correctly forwarded to the docker and ssh server. However, some display errors still can be faced reported while trying to run the GUI application. For testing the GUI inside the Docker container, run `xclock` and see if a window opens. The following guides are useful: https://www.baeldung.com/linux/forward-x-over-ssh, https://x410.dev/cookbook/enabling-ssh-x11-forwarding-in-visual-studio-code-for-remote-development/
- The application works only with SpatialDETR for now. Other models will be supported.

 