# Explainable ViT for 3D Object Detection from Multiple Cameras


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
To ease the setup of this project we provide a docker container and some convenience scripts (see `docker`). To setup use:
- (if not done alread) setup [Docker](https://docs.docker.com/engine/install/ubuntu/) and [Nvidia-Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- Run 
```bash
./docker/build.sh
```
to build the container
- If on a local machine, open 'bash /docker/run_loc.sh', otherwise open `/docker/run_cluster.sh` if on a remote server.
- Adapt the `nusc_data_dir` to the nuscenes directory and `work_dirs` to the direcory where model weights and pretrained weights directories are saved. Leave everything as it is if testing the project on the rhea3 cluster.

### Container Run and Setup
1. If on local machine run:
```bash
./docker/run_loc.sh
```
Otherwise, if working on a remote server, run:
```bash
./docker/run_cluster.sh
```
2. Now the container is running, but some packages still need to be installed. Run 
```bash
./docker/in_docker_setup.sh
```

### Application testing
1. After the docker container is set-up, run the application:
```bash 
python3 main.py
```
2. A GUI will appear. Select **File-Load model** to load the model configuration, the checkpoints and the dataset. Otherwise, if the same configurations are used each time, modify the **config.toml** file accordinly and select **File-Load from config file**.
3. Select a data index from the **data** menu, and click visualize. You can select the bounding box from the **Bounding boxes** menu.
4. Change the visualization settings with the drop-down menus. Then, click **Visualize**. 

### Train
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

### Issues and To-Do
- The shell for running the docker container makes sure that the display environment is correctly forwarded to the docker and ssh server. However, some display errors still can be faced reported while trying to run the GUI application. For testing the GUI inside the Docker container, run `xclock` and see if a window opens. The following guides are useful: https://www.baeldung.com/linux/forward-x-over-ssh, https://x410.dev/cookbook/enabling-ssh-x11-forwarding-in-visual-studio-code-for-remote-development/
- The application works both on Windows and Linux, but for now this README description is adapted only for Linux. I will describe how to configure the application for Windows too.
- The application works only with SpatialDETR for now. Other models will be supported.

 