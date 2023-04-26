## Setup
## Repository
1. Clone the repository `git clone https://github.com/ZassiM/SpatialDETR.git`

### Nuscenes Dataset
1. Follow the [mmdetection3d instructions](https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/datasets/nuscenes_det.html) to preprocess the data of the nuScenes dataset.

### Model weights
1. Download the [weights for FCOS3D](https://rwth-aachen.sciebo.de/s/asoSC5oMD1TNEsy) (which are used by SpatialDETR) and put them inside a directory called **pretrained**.
2. Download the [weights for SpatialDETR](https://rwth-aachen.sciebo.de/s/fgmMdPEQKQu9hz) (query_proj_value_proj) and put them inside a directory called **checkpoints**.


### Docker Setup
To ease the setup of this project we provide a docker container and some convenience scripts (see `docker`). To setup use:
- (if not done alread) setup [Docker](https://docs.docker.com/engine/install/ubuntu/) and [Nvidia-Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- Run `sudo ./docker/build.sh` to build the container
- If on a local machine, open `/docker/run_loc.sh`, otherwise run `/docker/run_ssh.sh` if on a remote server.
- Adapt the `nusc_data_dir` to the nuscenes directory and `work_dirs` to the direcory where model weights and pretrained weights directories are saved. Leave everything as it is if testing the project on the rhea3 cluster.

### Container Run and Setup
1. Run `./docker/run_loc.sh` if on local machine, otherwise run `./docker/run_ssh.sh` on remote server. 
2. Now the container is running, but some packages still need to be installed. Run `./docker/in_docker_setup.sh`, it will take about 5 minutes.

### Application testing
1. Run **python main.py**
2. A GUI will appear. Select **File-Load model** to load the model configuration, the checkpoints and the dataset. Otherwise, if the same configurations are used each time, modify the **config.toml** file accordinly and select **File-Load from config file**.
3. Select the **data index** and the **bounding box** to visualize. 
4. Change the visualization settings with the drop-down menus. Then, click **Visualize**. 


### Train
1. Use the configs in the **configs** folder to train SpatialDETR.  
For a baseline on a single gpu use:

`python ./mmdetection3d/tools/train.py configs/submission/frozen_4/query_proj_value_proj.py`  
  or for multi-gpu e.g. 4 gpus:  
`./mmdetection3d/tools/dist_train.sh configs/submission/frozen_4/query_proj_value_proj.py 4`

3. To test, use  
`./mmdetection3d/tools/dist_test.sh configs/submission/frozen_4/query_proj_value_proj.py /path/to/.pth 4 --eval=bbox`

### Issues
- The shell for running the docker container makes sure that the display environment is correctly forwarded to the docker and ssh server. However, some display errors still can be faced reported while trying to run the GUI application. For testing the GUI inside the Docker container, run `xclock` and see if a window opens. The following guides are useful: https://www.baeldung.com/linux/forward-x-over-ssh, https://x410.dev/cookbook/enabling-ssh-x11-forwarding-in-visual-studio-code-for-remote-development/
- The application works both on Windows and Linux, but for now this README description is adapted only for Linux. I will describe how to configure the application for Windows too.

 