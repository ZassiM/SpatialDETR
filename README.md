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
4. For Advanced Mode click **Settings-Advanced Mode**.

## Usage
The application provides a GUI with a series of dropdown menus for customization.

**Model Configuration**: Start by loading a SpatialDETR model configuration. You can opt for random sample data or choose a specific index.

**Prediction Threshold**: Set your prediction threshold. This helps to disregard low score queries and focus on confident predictions. The default threshold of 0.5 will streamline visualization by eliminating redundant queries.

**Explainability Methods**: Choose between Raw Attention, Grad-CAM, or Gradient Rollout to understand the model's decisions. Each method offers unique insights. With Raw Attention, you can select a specific head or fuse them using maximum, minimum, or mean. In Advanced Mode, you can also set a discard threshold to filter out lower attention maps, do perturbation or sanity tests.

**Saliency Maps Generation**: The app will then generate saliency maps—heat maps of the model's "attention"—across all six cameras for your chosen sample data. 

**Object-specific Analysis**: View saliency maps for all objects in a sample or focus on a specific one for detailed analysis. The app generates saliency maps for all layers and self-attention scores for the selected object, helping you understand the detection process layer-by-layer.

**Real-time Visualization**: You can generate and view a sequence of images with saliency maps for all objects, giving a sense of the attention mechanism's dynamics. Pause the sequence, select objects for further analysis, and resume. Apply an object-specific filter to focus on samples containing a certain object.

**Visualization Options**: Visualize a Bird's Eye View (BEV) perspective using LiDAR points and bounding boxes for a broad environmental view. You can convert 3D bounding boxes into 2D for simpler visualization and overlay ground truth bounding boxes on model predictions to compare performance. You can also create a segmentation mask from the saliency map, helpful when comparing with the dataset's ground truth.

Overview of the GUI application:
![](misc/ui-overview.png "Overview of the GUI application")  

Figure showing saliency maps of a car through the layers:
![](misc/ui-crossattn.png "Saliency map of a car through the layers")  

Figure illustrating the self attention of a car with other objects in the scene:
![](misc/ui-selfattn.png "Self attention of a car with other objects in the scene")  

## Issues and To-Do
- The shell for running the docker container makes sure that the display environment is correctly forwarded to the docker and ssh server. However, some display errors still can be faced reported while trying to run the GUI application. For testing the GUI inside the Docker container, run `xclock` and see if a window opens. The following guides are useful: https://www.baeldung.com/linux/forward-x-over-ssh, https://x410.dev/cookbook/enabling-ssh-x11-forwarding-in-visual-studio-code-for-remote-development/
- The application works only with SpatialDETR.

 