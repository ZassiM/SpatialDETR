# Explainable Transformer-based 3D Object Detector from multiple cameras
This application assists in understanding the reasoning of a 3D Object detector, SpatialDETR. The Machine Learning framework used is Pytorch. The Graphical User Interface is created with Tkinter. It uses the MMDetection3D framework to load a model and dataset. The dataset used for this application is Nuscenes.
The application displays saliency maps from various explainability techniques, allowing you to see where the model is focusing at. It also provides additional features such as visualization in Birth Eye View (BEV), generation of a segmentation map from the saliency map and visualization of the objects self-attention mechanism.

## Setup
### Repository
1. Clone the repository together with its submodules: 

```bash
git clone --recurse-submodules https://gitlab.ika.rwth-aachen.de/ma-zahr/xai.git 
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
python scripts/main.py
```
2. A GUI will appear. Select **File-Load model** to load the model configuration and the checkpoints. Otherwise, if the same configurations are used each time, modify the **config.toml** file accordinly and select **File-Load from config file**.
3. Change the visualization settings with the drop-down menus. Then, click **Visualize**. 

## Usage
The application provides a GUI with a series of dropdown menus for customization.

1. **Change Prediction Threshold**: This filters out queries with scores below the chosen threshold. By default, the threshold is set at 0.5.
2. **Select Object for Explainability Map**: This allows you to select specific objects for individual saliency map visualization.
3. **Select Explainability Technique**: Choose between Raw Attention, Grad-CAM, and Gradient Rollout. For raw attention, there are further options to select a particular head, or to fuse them by taking the maximum, minimum, or mean. You can also adjust the discard threshold.
4. **Objects Self-Attention Visualization**: Shows the attention score between the selected object query and all other objects in the images.
5. **Conversion of 3D to 2D Bounding Boxes**: This will convert the bounding boxes from LiDAR to camera coordinates, which are easier to visualize on top of a 2D bounding box.
6. **Use Otsu's Thresholding Method**: This will generate a segmentation mask from the saliency map.
7. **Generate Video**: Allows the generation of a video-like visualization of a scene. The generated sequence can be saved as a collection of images or can be visualized directly on the UI. During the video, you can pause, select an object for explainability, and then resume the video.

This figure shows how the application looks like when visualizing the saliency maps of a truck for every layer of the model. The charts show the self-attention contributions for each object.

![](misc/readme_overview.png "Overview of the GUI application")  

## Issues and To-Do
- The shell for running the docker container makes sure that the display environment is correctly forwarded to the docker and ssh server. However, some display errors still can be faced reported while trying to run the GUI application. For testing the GUI inside the Docker container, run `xclock` and see if a window opens. The following guides are useful: https://www.baeldung.com/linux/forward-x-over-ssh, https://x410.dev/cookbook/enabling-ssh-x11-forwarding-in-visual-studio-code-for-remote-development/
- The application works only with SpatialDETR.

 