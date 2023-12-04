import tkinter as tk
import numpy as np
import torch
import cv2
import mmcv
import os
import random
import pickle
import shutil
from tkinterhtml import HtmlFrame
import markdown
import codecs
from tkinter import filedialog as fd
from mmcv.parallel import DataContainer as DC
from tkinter.messagebox import showinfo
from tkinter import scrolledtext
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from modules.Configs import Configs
from modules.Explainability import ExplainableTransformer
from modules.Model import Model
from PIL import ImageGrab

class BaseApp(tk.Tk):
    '''
    Application User Interface
    '''
    def __init__(self):
        '''
        Tkinter initialization with model loading option.
        '''
        super().__init__()

        # Tkinter-related settings
        self.tk.call("source", "misc/theme/azure.tcl")
        self.tk.call("set_theme", "light")
        self.title('Explainable Multi-Sensor 3D Object Detection with Transformers')
        self.geometry('1500x1500')
        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.bg_color = self.option_get('background', '*')      # Background color changes when theme is changed

        # Canvas-related settings for plotting on Tkinter window
        self.canvas, self.fig, self.spec = None, None, None
        
        self.started_app = False        # True when model & dataset are loaded
        self.advanced_mode = False      # True when advanced mode is selected
        self.video_gen_bool = False     # True when a video has been generated
        self.video_loaded = False       # True when a video has been generated
        self.layers_video = 0           # Number of layers from the loaded video
        self.img_labels = None          # Object labels from the loaded video

        # Load/create txt file containing saved dataset indices
        self.indices_file = 'misc/indices.txt'
        if not os.path.exists(self.indices_file):
            open(self.indices_file, 'w').close()

        # Load scenes information from NuScenes. Used when visualizing video
        self.scene_samples = []
        self.scene_descriptions = []
        with open('misc/scenes.txt', 'r') as file:
            for line in file:
                parts = line.split('|')
                if len(parts) >= 2:
                    number = int(parts[0].strip().split()[-1])
                    sentence = parts[1].strip()
                    self.scene_samples.append(number)
                    self.scene_descriptions.append(sentence)

        # Used while plotting the images from the 6 camers
        self.cam_idx = [2, 0, 1, 5, 3, 4]      
        self.cam_names = ["Front-left", "Front", "Front-right", "Back-left", "Back", "Back-right"]
        self.bbox_coords, self.saliency_maps_objects = [], []
        
        # Main Tkinter menu in which all other cascade menus are added
        self.menubar = tk.Menu(self)

        # Cascade menus for loading model, selecting the GPU and screenshot
        self.config(menu=self.menubar)
        file_opt, gpu_opt, screenshot_opt = tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.gpu_id = tk.IntVar()
        self.gpu_id.set(0)
        self.screenshot_cam = tk.IntVar()
        self.screenshot_cam.set(-1)
        file_opt.add_command(label=" Load model", command=self.load_model)
        file_opt.add_command(label=" Load model from config file", command=lambda: self.load_model(from_config=True))
        file_opt.add_command(label=" Save index", command=lambda: self.insert_entry(type=1))
        file_opt.add_cascade(label=" Screenshot", menu=screenshot_opt)
        file_opt.add_cascade(label=" Gpu", menu=gpu_opt)
        file_opt.add_separator()
        file_opt.add_command(label=" Show car setup", command=self.show_car)
        file_opt.add_command(label=" How to use", command=self.show_app_info)
        
        # Adding all possible GPUs available
        message = "You need to reload the model to apply GPU change."
        for i in range(torch.cuda.device_count()):
            gpu_opt.add_radiobutton(label=f"GPU {i}", variable=self.gpu_id, value=i, command=lambda: self.show_message(message))

        # Ability to select which camera to use for taking a screenshot
        screenshot_opt.add_radiobutton(label=f"Full", variable=self.screenshot_cam, value=-1, command=self.capture)
        for i, cam_name in zip(self.cam_idx, self.cam_names):
            screenshot_opt.add_radiobutton(label=f"{cam_name} camera", variable=self.screenshot_cam, value=i, command=self.capture)

        # The complete File menu
        self.menubar.add_cascade(label=" File", menu=file_opt)

    def load_model(self, from_config=False):
        ''' Loading of the model and dataset from cfg file into the ObjectDetector object.'''
        self.ObjectDetector = Model()

        if not from_config:
            self.ObjectDetector.load_model(gpu_id=self.gpu_id.get())
        else:
            self.ObjectDetector.load_from_config()

        # ExplainableTransformer object used through the app
        self.ExplainableModel = ExplainableTransformer(self.ObjectDetector)

        # Synced configurations: when a value is changed, the triggered function is called
        data_configs, expl_configs = [], []
        self.data_configs = Configs(data_configs, triggered_function=self.update_data, type=0)
        self.expl_configs = Configs(expl_configs, triggered_function=self.ExplainableModel.generate_explainability, type=1)

        # This initializes all the other Tkinter menus
        if not self.started_app:
            print("Starting app...")
            self.start_app()   
            self.random_data_idx()
            self.started_app = True
            print("Completed.\n")

        # Updates the information text each time a new model is loaded
        self.update_info_label()

    def start_app(self):
        '''
        It starts the UI after loading the model by adding all other menus. Variables are initialized.
        '''
        self.frame = tk.Frame(self)
        self.frame.pack(fill=tk.Y)
        self.info_text = tk.StringVar()
        self.info_label = tk.Label(self.frame, textvariable=self.info_text, anchor=tk.CENTER)
        self.info_label.bind("<Button-1>", lambda event: self.show_model_info())
        self.info_label.bind("<Enter>", lambda event: self.red_text())  # Text becomes red while hovering on top of it
        self.info_label.bind("<Leave>", lambda event: self.black_text())
        self.info_label.pack(side=tk.TOP)

        self.info_text_video = tk.StringVar()
        self.info_label_video = tk.Label(self.frame, textvariable=self.info_text_video, anchor=tk.CENTER)

        # ***Main Menu*** for Data settings
        self.dataidx_opt, self.select_idx_opt, self.thr_opt = tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.selected_threshold = tk.DoubleVar()
        self.selected_threshold.set(0.5)
        values = np.arange(0.0, 1, 0.1).round(1)
        for i in values:
            self.thr_opt.add_radiobutton(label=i, variable=self.selected_threshold)

        self.select_idx_opt.add_command(label="         Insert index", command=lambda: self.insert_entry(type=0))
        self.selected_data_idx = tk.IntVar()
        with open(self.indices_file, 'r') as file:
            for line in file:
                self.select_idx_opt.add_radiobutton(label=line.strip(), variable=self.selected_data_idx, command=self.update_idx, value=line.split()[0])
        self.dataidx_opt.add_cascade(label=" Select sample index", menu=self.select_idx_opt)
        self.dataidx_opt.add_command(label=" Select random sample", command=self.random_data_idx)
        self.dataidx_opt.add_separator()
        self.dataidx_opt.add_command(label=" Show LiDAR", command=self.show_lidar)

        # ***Main Menu*** for Video settings
        videolength_opt = tk.Menu(self.menubar)
        video_lengths = np.arange(0, 1200, 100)
        video_lengths[0] = 10
        video_lengths[-1] = len(self.ObjectDetector.dataset)
        self.video_length = tk.IntVar()
        self.video_length.set(video_lengths[0])
        for i in range(len(video_lengths)):
            videolength_opt.add_radiobutton(label=video_lengths[i], variable=self.video_length , value=video_lengths[i])

        # ***Advanced Options*** Select a particular class of objects during video visualization
        self.filter_opt = tk.Menu(self.menubar)
        self.selected_filter = tk.IntVar()
        for label, class_name in enumerate(self.ObjectDetector.class_names):
            self.filter_opt.add_radiobutton(label=class_name.replace("_", " ", 1).capitalize(), variable=self.selected_filter, value=label, command=self.update_object_filter)
        self.filter_opt.add_radiobutton(label="All", variable=self.selected_filter, value=label+1,  command=self.update_object_filter)
        self.selected_filter.set(label+1)

        # ***Advanced Options*** Select a particular video delay between frames
        self.delay_opt = tk.Menu(self.menubar)
        video_delays = np.arange(0, 35, 5)
        video_delays[0] = 1
        self.video_delay = tk.IntVar()
        self.video_delay.set(video_delays[0])
        for i in range(len(video_delays)):
            self.delay_opt.add_radiobutton(label=video_delays[i], variable=self.video_delay, value=video_delays[i])
        
        self.aggregate_layers = tk.BooleanVar()    
        self.aggregate_layers.set(True)
        self.video_opt = tk.Menu(self.menubar)
        self.video_opt.add_command(label=" Generate", command=self.generate_video)
        self.video_opt.add_command(label=" Load", command=self.load_video)
        self.video_opt.add_cascade(label=" Video length", menu=videolength_opt)
        self.video_opt.add_command(label=" Select scene", command=lambda: self.insert_entry(type=2))

        # ***Main Menu*** for object selection
        self.bbox_opt = tk.Menu(self.menubar)
        self.single_bbox = tk.BooleanVar()
        self.select_all_bboxes = tk.BooleanVar()
        self.select_all_bboxes.set(True)
        self.bbox_opt.add_checkbutton(label=" Single object", onvalue=1, offvalue=0, variable=self.single_bbox, command=self.single_bbox_select) 
        self.bbox_opt.add_checkbutton(label=" Select all", onvalue=1, offvalue=0, variable=self.select_all_bboxes, command=self.initialize_bboxes)
        self.bbox_opt.add_separator()

        # ***Main Menu*** for Explainable options
        self.expl_opt = tk.Menu(self.menubar)
        self.raw_attention_menu, self.grad_cam_menu, self.grad_rollout_menu = tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.expl_options = ["Raw Attention", "Grad-CAM", "Gradient Rollout", "Self Attention"]

        # Explainable mechanism selection
        expl_type_opt = tk.Menu(self.menubar)
        self.expl_opt.add_cascade(label="Mechanism", menu=expl_type_opt)
        self.selected_expl_type = tk.StringVar()
        self.selected_expl_type.set(self.expl_options[0])
        for opt in self.expl_options:
            expl_type_opt.add_radiobutton(
                label=opt,
                variable=self.selected_expl_type,
                value=opt,
                command=self.update_info_label
            )

        # ***Advanced Options*** Head Fusion Selection
        self.head_fusion_options = ["max", "min", "mean", "zero_clamp_mean"]
        self.selected_head_fusion = tk.StringVar()
        self.selected_head_fusion.set(self.head_fusion_options[0])

        head_fusion_opt = tk.Menu(self.menubar)
        for opt in self.head_fusion_options:
            head_fusion_opt.add_radiobutton(
                label=opt.capitalize(),
                variable=self.selected_head_fusion,
                value=opt,
                command=self.update_info_label
            )
        for head in range(self.ObjectDetector.num_heads):
            head_fusion_opt.add_radiobutton(
                label=str(head),
                variable=self.selected_head_fusion,
                value=str(head)
            )

        self.raw_attention_menu.add_cascade(label=" Head Fusion", menu=head_fusion_opt)

        # ***Advanced Options*** Layer Fusion Selection
        self.layer_fusion_options = ["max", "min", "mean", "last", "zero_clamp_mean"]
        layer_fusion_opt = tk.Menu(self.menubar)
        self.selected_layer_fusion_type = tk.StringVar()
        self.selected_layer_fusion_type.set(self.layer_fusion_options[0])
        for opt in self.layer_fusion_options:
            layer_fusion_opt.add_radiobutton(
                label=opt.capitalize(),
                variable=self.selected_layer_fusion_type,
                value=opt,
                command=self.update_info_label
            )

        self.raw_attention_menu.add_cascade(label=" Layer Fusion", menu=layer_fusion_opt)

        # ***Advanced Options*** Grad-CAM
        self.grad_cam_types = ["default"]
        self.selected_gradcam_type = tk.StringVar()
        self.selected_gradcam_type.set(self.grad_cam_types[0])
        for i in range(len(self.grad_cam_types)):
            self.grad_cam_menu.add_radiobutton(label=self.grad_cam_types[i].capitalize(), variable=self.selected_gradcam_type, value=self.grad_cam_types[i])

        # ***Advanced Options*** Gradient Rollout
        self.handle_residual, self.apply_rule = tk.BooleanVar(), tk.BooleanVar()
        self.handle_residual.set(True)
        self.apply_rule.set(True)
        self.grad_rollout_menu.add_checkbutton(label=" Handle residual", variable=self.handle_residual, onvalue=1, offvalue=0)
        self.grad_rollout_menu.add_checkbutton(label=" Apply rule", variable=self.apply_rule, onvalue=1, offvalue=0)

        # ***Advanced Options*** Perturbation Menu
        pert_step_opt, pert_type_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.selected_pert_step = tk.DoubleVar()
        self.selected_pert_step.set(-1)
        self.selected_pert_type = tk.StringVar()
        pert_mask_types = ["Mask", "Blur"]
        self.selected_pert_type.set(pert_mask_types[0])
        pert_steps = np.arange(0, 1.05, 0.05)
        for step in pert_steps:
            pert_step_opt.add_radiobutton(label=f"{int(step*100)} %",
                variable=self.selected_pert_step, value=step)
        for mask_type in pert_mask_types:
            pert_type_opt.add_radiobutton(label=mask_type,
                variable=self.selected_pert_type, value=mask_type)

        # ***Advanced Options*** Positive Negative Perturbation
        pert_pos_neg_opt = tk.Menu(self.menubar)
        self.selected_pert_pos_neg = tk.StringVar()
        self.selected_pert_pos_neg.set("Positive")
        pert_pos_neg_opt.add_radiobutton(label="Positive",
            variable=self.selected_pert_pos_neg, value="Positive")
        pert_pos_neg_opt.add_radiobutton(label="Negative",
            variable=self.selected_pert_pos_neg, value="Negative")

        # ***Advanced Options*** Visualization Colour of Perturbation
        pert_colour_opt = tk.Menu(self.menubar)
        self.selected_pert_colour = tk.StringVar()
        self.selected_pert_colour.set("mean")
        pert_colour_opt.add_radiobutton(label="Mean",
            variable=self.selected_pert_colour, value="mean")
        pert_colour_opt.add_radiobutton(label="Red",
            variable=self.selected_pert_colour, value="red")

        # ***Advanced Options*** Perturbation Menu
        self.pert_opt = tk.Menu(self.menubar)
        self.pert_opt.add_cascade(label="Step", menu=pert_step_opt)
        self.pert_opt.add_cascade(label="Type", menu=pert_type_opt)
        self.pert_opt.add_cascade(label="PosNeg", menu=pert_pos_neg_opt)
        self.pert_opt.add_cascade(label="Colour", menu=pert_colour_opt)

        # ***Advanced Options*** Sanity Check
        self.sancheck_opt = tk.Menu(self.menubar)
        self.selected_sancheck_layer = []
        for layer in range(self.ObjectDetector.num_layers):
            var = tk.IntVar()
            self.selected_sancheck_layer.append(var)
            self.sancheck_opt.add_checkbutton(label=layer,  onvalue=1, offvalue=0, variable=var)

        # ***Advanced Options*** Discard ratio for attention weights
        self.dr_opt, self.int_opt, self.beta_opt = tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.gen_segmentation = tk.BooleanVar()
        self.selected_discard_threshold = tk.DoubleVar()
        self.selected_intensity = tk.IntVar()
        self.selected_beta = tk.DoubleVar()
        discard_ratios = np.arange(0.0, 1, 0.1).round(1)
        intensities = np.arange(200, 270, 10)
        betas = np.round(np.arange(0.1, 1.1, 0.1), 1)
        intensities[-1] = 255
        self.selected_discard_threshold.set(discard_ratios[3])
        self.selected_intensity.set(intensities[3])
        self.selected_beta.set(betas[6])
        for i in discard_ratios:
            self.dr_opt.add_radiobutton(label=i, variable=self.selected_discard_threshold)
        for i in intensities:
            self.int_opt.add_radiobutton(label=i, variable=self.selected_intensity)
        for i in betas:
            self.beta_opt.add_radiobutton(label=i, variable=self.selected_beta)

        quality_opt = tk.Menu(self.menubar)
        map_qualities = ["Low", "Medium", "High"]
        self.selected_map_quality = tk.StringVar()
        self.selected_map_quality.set(map_qualities[2])
        for i in range(len(map_qualities)):
            quality_opt.add_radiobutton(label=map_qualities[i], variable=self.selected_map_quality, value=map_qualities[i])

        # ***Main Menu*** for additional settings
        add_settings = tk.Menu(self.menubar)
        self.GT_bool, self.overlay_bool, self.bbox_2d, self.capture_object, self.remove_pad, self.draw_bboxes = \
            tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.overlay_bool.set(True)
        self.remove_pad.set(True)
        self.draw_bboxes.set(True)
        add_settings.add_checkbutton(label=" Show predicted OBB", onvalue=1, offvalue=0, variable=self.draw_bboxes)
        add_settings.add_checkbutton(label=" Show saliency maps", onvalue=1, offvalue=0, variable=self.overlay_bool)
        add_settings.add_checkbutton(label=" 2D OBB", onvalue=1, offvalue=0, variable=self.bbox_2d)
        add_settings.add_checkbutton(label=" Show Ground Truth OBB", onvalue=1, offvalue=0, variable=self.GT_bool)
        add_settings.add_checkbutton(label=" Capture saliency maps", onvalue=1, offvalue=0, variable=self.capture_object)
        add_settings.add_command(label=" Change theme", command=self.change_theme)
        add_settings.add_separator()
        add_settings.add_checkbutton(label=" Advanced Mode", command=self.toggle_advanced_mode)

        # Adding all menus to the main maenubar
        self.add_separator()
        self.menubar.add_cascade(label="Data", menu=self.dataidx_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Video", menu=self.video_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Objects", menu=self.bbox_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Explainability", menu=self.expl_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Settings", menu=add_settings)
        self.add_separator("|")
        self.menubar.add_command(label="Visualize", command=self.visualize)
        self.add_separator("|")

    def toggle_advanced_mode(self):
        ''' Gives more flexibility to the user/developer for exploring XAI. '''
        if not self.advanced_mode:
            self.dataidx_opt.insert_cascade(2, label=" Select prediction threshold", menu=self.thr_opt)

            self.video_opt.add_cascade(label=" Video delay", menu=self.delay_opt)
            self.video_opt.add_cascade(label=" Filter object", menu=self.filter_opt)
            self.video_opt.add_checkbutton(label=" Aggregate layers", onvalue=1, offvalue=0, variable=self.aggregate_layers)
            self.expl_opt.add_separator()
            self.expl_opt.add_cascade(label=self.expl_options[0], menu=self.raw_attention_menu)
            self.expl_opt.add_cascade(label=self.expl_options[1], menu=self.grad_cam_menu)
            self.expl_opt.add_cascade(label=self.expl_options[2], menu=self.grad_rollout_menu)
            self.expl_opt.add_separator()
            self.expl_opt.add_cascade(label="Perturbation", menu=self.pert_opt)
            self.expl_opt.add_cascade(label="Sanity check", menu=self.sancheck_opt)
            self.expl_opt.add_cascade(label="Discard threshold", menu=self.dr_opt)
            self.expl_opt.add_cascade(label="Saliency map intensity", menu=self.int_opt)
            self.expl_opt.add_checkbutton(label="Generate segmentation map", onvalue=1, offvalue=0, variable=self.gen_segmentation)

            self.title('Explainable Multi-Sensor 3D Object Detection with Transformers   |   ADVANCED MODE')
            self.advanced_mode = True

        else:
            # Deactivate Advanced Mode
            self.dataidx_opt.delete(2)

            end_idx = self.video_opt.index('end')
            self.video_opt.delete(end_idx-2, end_idx)
            end_idx = self.expl_opt.index('end')
            self.expl_opt.delete(1, end_idx)
            self.title('Explainable Multi-Sensor 3D Object Detection with Transformers')
            self.advanced_mode = False
        
        self.update_info_label()

    def show_car(self):
        ''' Shows the sensor setup of the car used by nuScenes.'''
        image_window = tk.Toplevel()
        image_path = 'misc/car.jpg'
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(image_window, image=photo)
        label.pack()
        image_window.mainloop()
        
    def update_idx(self):
        ''' Updates the selected dataset index and the info label. '''
        self.data_idx = self.selected_data_idx.get()
        self.update_info_label()

    def open_md_file(self):
        ''' Opens README file in the markdown format. '''
        input_file = codecs.open("README.md", mode="r", encoding="utf-8")
        text = input_file.read()
        html = markdown.markdown(text)
        return html

    def show_app_info(self):
        ''' Visualizes the README file. '''
        top = tk.Toplevel(self)
        frame = HtmlFrame(top, horizontal_scrollbar="auto")
        frame.grid(sticky=tk.NSEW)
        frame.set_content(self.open_md_file())

    def random_data_idx(self):
        ''' Generates a random index for the dataset. '''
        idx = random.randint(0, len(self.ObjectDetector.dataset)-1)
        self.data_idx = idx
        self.selected_pert_step.set(-1)
        self.update_info_label()

    def update_data(self, gradients=False, initialize_bboxes=True):
        ''' Extract data, update required parameters'''

        # Load selected data from dataloader, manual DataContainer fixes are needed
        data = self.ObjectDetector.dataset[self.data_idx]
        metas = [[data['img_metas'][0].data]]
        img = [data['img'][0].data.unsqueeze(0)]
        data['img_metas'][0] = DC(metas, cpu_only=True)
        data['img'][0] = DC(img)
        self.data = data

        if "points" in self.data.keys():
            self.data.pop("points")

        # Attention scores are extracted, together with gradients if grad-CAM is selected
        if gradients:
            outputs = self.ExplainableModel.extract_attentions(self.data, self.bbox_idx)
        else:
            outputs = self.ExplainableModel.extract_attentions(self.data)

        # Those are needed to index the bboxes decoded by the NMS-Free decoder
        self.nms_idxs = self.ObjectDetector.model.module.pts_bbox_head.bbox_coder.get_indexes()
        self.bbox_scores = self.ObjectDetector.model.module.pts_bbox_head.bbox_coder.get_scores()
        self.outputs = outputs[0]["pts_bbox"]
        self.thr_idxs = self.outputs['scores_3d'] > self.selected_threshold.get()
        self.labels = self.outputs['labels_3d'][self.thr_idxs]
        self.pred_bboxes = self.outputs["boxes_3d"][self.thr_idxs]
        
        # Verify if no objects are detected
        self.no_object = False
        if len(self.labels) == 0:
            self.no_object = True
            if not self.video_gen_bool:
                print("No object detected.")
                self.show_message("No object detected")

        # Extract image metas which contain, for example, the lidar to camera projection matrices
        self.img_metas = self.data["img_metas"][0]._data[0][0]
        self.data_description = None
        self.color_dict = None

        if self.selected_pert_step.get() > 0:
            print("Perturbating images...")
            xai_maps = self.ExplainableModel.xai_maps
            img = img[0][0]
            img = img[:, :, :self.ObjectDetector.ori_shape[0], :self.ObjectDetector.ori_shape[1]]  # [num_cams x height x width x channels]

            img_pert_list = []
            for camidx in range(len(xai_maps)):
                img_pert = img[camidx].permute(1, 2, 0).numpy()
                xai_cam = xai_maps[camidx]
                
                if self.selected_pert_pos_neg.get() == "Negative":
                    print("Using Negative Perturbation")
                    xai_cam = -xai_cam
                else:
                    print("Using Positive Perturbation")

                num_pixels_removed = int(self.selected_pert_step.get() * xai_cam.numel())
                print("Number of Pixel Removed for Cam {1}: {0}".format(num_pixels_removed, camidx))
                _, indices = torch.topk(xai_cam.flatten(), num_pixels_removed)
                row_indices, col_indices = indices // xai_cam.size(1), indices % xai_cam.size(1)

                if self.selected_pert_type.get() == "Mask":
                    img_pert[row_indices, col_indices] = np.mean(img_pert, axis=(0, 1)) if self.selected_pert_colour.get() == "mean" else (-97.53, -114, 131.325)
                elif self.selected_pert_type.get() == "Blur":
                    blur_section  = img_pert[row_indices, col_indices]
                    blurred_section = cv2.GaussianBlur(blur_section, (9, 9), 0)
                    img_pert[row_indices, col_indices] = blurred_section

                img_pert_list.append(img_pert)

            if len(img_pert_list) > 0:
                # Save_img the perturbed 6 camera images into the data input
                img_pert_list = torch.from_numpy(np.stack(img_pert_list))
                img = [img_pert_list.permute(0, 3, 1, 2).unsqueeze(0)] # img = [torch.Size([1, 6, 3, 928, 1600])
                self.data['img'][0] = DC(img)

        # Extract the 6 camera images from the data and remove the padded pixels
        imgs = self.data["img"][0]._data[0].numpy()[0]
        imgs = imgs.transpose(0, 2, 3, 1)
        if self.remove_pad.get():
            imgs = imgs[:, :self.ObjectDetector.ori_shape[0], :self.ObjectDetector.ori_shape[1], :]  # [num_cams x height x width x channels]
        
        # Get mean std
        img_norm_cfg = self.ObjectDetector.cfg.get('img_norm_cfg')
        mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
        std = np.array(img_norm_cfg["std"], dtype=np.float32)

        # Denormalize the images
        for i in range(len(imgs)):
            imgs[i] = mmcv.imdenormalize(imgs[i], mean, std, to_bgr=False)
        self.imgs = imgs.astype(np.uint8)

        # By default, when new data is loadedm all objects are selected for visualizing all bounding boxes
        all_select = True
        if self.selected_pert_step.get() > 0:
            all_select = False
        if (initialize_bboxes and not self.video_gen_bool):
            self.update_objects_list(all_select=all_select)

    def add_separator(self, sep=""):
        self.menubar.add_command(label=sep, activebackground=self.menubar.cget("background"))
        # sep="\u22EE"

    def show_message(self, message):
        showinfo(title=None, message=message)

    def red_text(self, event=None):
        self.info_label.config(fg="red")

    def black_text(self, event=None):
        if self.tk.call("ttk::style", "theme", "use") == "azure-dark":
            self.info_label.config(fg="white")
        else:
            self.info_label.config(fg="black")

    def show_model_info(self, event=None):
        ''' Show information about model when clicking on the label. '''
        popup = tk.Toplevel(self)
        popup.geometry("700x1000")
        popup.title(f"Model {self.ObjectDetector.model_name}")

        text = scrolledtext.ScrolledText(popup, wrap=tk.WORD)
        for k, v in self.ObjectDetector.model.module.__dict__["_modules"].items():
            text.insert(tk.END, f"{k.upper()}\n", 'key')
            text.insert(tk.END, f"{v}\n\n")
            text.tag_config('key', background="yellow", foreground="red")

        text.pack(expand=True, fill='both')
        text.configure(state="disabled")

    def insert_entry(self, type=0):
        ''' Inserting input from user. '''
        popup = tk.Toplevel(self)
        popup.geometry("80x50")

        self.entry = tk.Entry(popup)
        self.entry.pack(fill=tk.BOTH, expand=True)

        button = tk.Button(popup, text="OK", command=lambda: self.close_entry(popup, type))
        button.pack()

    def close_entry(self, popup, type):
        ''' Type 0: data index; type 1: save data index; type 2: scene selection. '''
        entry = self.entry.get()
        if type == 0:
            if entry.isnumeric() and int(entry) <= (len(self.ObjectDetector.dataset)-1):
                self.data_idx = int(entry)
                self.update_info_label()
            else:
                self.show_message(f"Insert an integer between 0 and {len(self.ObjectDetector.dataset)}")
                return
        elif type == 1:
            self.data_description = f'{self.data_idx} | {entry}'
            self.select_idx_opt.add_radiobutton(label=self.data_description, variable=self.selected_data_idx, command=self.update_idx, value=self.data_idx)
            with open(self.indices_file, 'a') as file:
                file.write(f'{self.data_description}\n')
            with open(self.indices_file, 'r') as file:
                lines = file.readlines()
            lines.sort(key=lambda x: int(x.split(' | ')[0]))
            with open(self.indices_file, 'w') as file:
                file.writelines(lines)
        elif type == 2:
            if entry.isnumeric() and int(entry) <= (len(self.scene_samples) - 1):
                self.video_scene = int(entry)
                self.data_idx = 0
                for i in range(self.video_scene):
                    self.data_idx += self.scene_samples[i]
                self.video_length.set(self.scene_samples[self.video_scene])
                self.scene_description = self.scene_descriptions[self.video_scene]
            else:
                self.show_message(f"Insert an integer between 0 and {len(self.scene_samples) -1}")
                return
        
        popup.destroy()

    def update_info_label(self, info=None, idx=None):
        ''' Updates the info text each time a value changes. '''
        if idx is None:
            idx = self.data_idx
        if info is None:
            info = f'Model: {self.ObjectDetector.model_name} | '\
                   f'Dataset: {self.ObjectDetector.dataset_name} | '\
                   f'Sample index: {idx} | '\
                   f'Mechanism: {self.selected_expl_type.get()}'
            if self.advanced_mode and self.selected_expl_type.get() == "Raw Attention":
                info += f'| Head Fusion: {self.selected_head_fusion.get().capitalize()} | '\
                        f'Layer Fusion: {self.selected_layer_fusion_type.get().capitalize()}'
            if self.video_gen_bool:
                if self.layers_video > 1:
                    info += f" | Layer {self.layer_idx}"
        self.info_text.set(info)

    def update_info_video_label(self):
        ''' Updated the info while visualizing a video scene. '''
        text = "KEYLEFT: back, KEYRIGHT: forward, SPACE: pause/resume"
        if self.layers_video > 1:
            text += ", KEYUP: previous layer, KEYDOWN: next layer"
        if hasattr(self, "scene_description"):
            text += f"\n{self.scene_description}"
        self.info_text_video.set(text)

    def update_objects_list(self, labels=None, single_select=False, all_select=True):
        ''' Updates the object menu with the objects present in the data. '''
        if labels is None:
            labels = self.labels
        self.bboxes = []
        self.bbox_opt.delete(3, 'end')
        for i in range(len(labels)):
            view_bbox = tk.BooleanVar()
            view_bbox.set(False)
            self.bboxes.append(view_bbox)
            object_class = self.ObjectDetector.class_names[labels[i].item()].replace("_", " ", 1).capitalize()
            self.bbox_opt.add_checkbutton(label=f" {i}: {object_class}", onvalue=1, offvalue=0, variable=self.bboxes[i], command=lambda idx=i: self.single_bbox_select(idx))

        self.initialize_bboxes(single_select, all_select)

    def initialize_bboxes(self, single_select=False, all_select=False):
        ''' Initializes the selected objects depending on the user. '''
        if all_select:
            self.select_all_bboxes.set(True)
        if single_select:
            self.single_bbox.set(True)
            self.select_all_bboxes.set(False)
        if hasattr(self, "bboxes"):
            if self.select_all_bboxes.get():
                self.single_bbox.set(False)
                for i in range(len(self.bboxes)):
                    self.bboxes[i].set(True)
            else:
                if len(self.bboxes) > 0:
                    self.bboxes[0].set(True)

    def single_bbox_select(self, idx=None, single_select=False):
        ''' Selects the first object and deselect all others. '''
        self.select_all_bboxes.set(False)
        if single_select:
            self.single_bbox.set(True)
        if self.single_bbox.get():
            if idx is None:
                idx = 0
            for i in range(len(self.bboxes)):
                if i != idx:
                    self.bboxes[i].set(False)

    def get_object_camera(self):
        ''' Extracts the camera index with the highest score for a selected object. '''
        scores = self.ExplainableModel.scores
        if not scores:
            return -1
        cam_obj = np.argmax(scores)
        return cam_obj

    def capture(self, event=None):
        ''' Takes screenshot of the whole window or of a selected camera. '''
        screenshots_path = "screenshots/cameras/"

        if not os.path.exists(screenshots_path):
            os.makedirs(screenshots_path)

        if self.data_description:
            path = screenshots_path + self.data_description.replace(" | ", "_").replace(" ", "_") + ".png"
        else:
            expl_string = self.selected_expl_type.get().replace(" ", "_")
            path = screenshots_path + f"{self.ObjectDetector.model_name}_{expl_string}_{self.data_idx}"
            if self.selected_pert_step.get() != -1:
                path += f"_p{int(self.selected_pert_step.get()*100)}"
        
        if self.screenshot_cam.get() == -1:
            # Whole window screenshot
            path += ".png"
            self.fig.savefig(path, dpi=300, transparent=True)
        else:
            # Particular camera view screenshot
            fig_save, ax_save = plt.subplots()
            ax_save.imshow(self.cam_imgs[self.screenshot_cam.get()])
            ax_save.axis('off')  # Turn off axis

            path += "_" + self.cam_names[self.cam_idx.index(self.screenshot_cam.get())] + ".png"
            fig_save.savefig(path, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close(fig_save)    
            
        print(f"Screenshot saved in {path}.")

    def change_theme(self):
        ''' Toggle theme between light and dark. '''
        if self.tk.call("ttk::style", "theme", "use") == "azure-dark":
            self.tk.call("set_theme", "light")
            self.bg_color = "white"
        else:
            self.tk.call("set_theme", "dark")
            self.bg_color = self.option_get('background', '*')

        self.fig.set_facecolor(self.bg_color)
        self.canvas.draw()

    def generate_saliency_map(self, img, xai_map):
        ''' Generates saliency map by blending an XAI map onto the image, '''
        xai_map_colored = cv2.applyColorMap(np.uint8(xai_map * self.selected_intensity.get()), cv2.COLORMAP_TURBO)
        xai_map_colored = cv2.resize(xai_map_colored, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        xai_map_colored = np.float32(xai_map_colored)

        # normalize xai_map to create an alpha mask with values in range 0-1
        xai_map_normalized = cv2.normalize(np.uint8(xai_map * self.selected_intensity.get()), None, alpha=0, beta=self.selected_beta.get(), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        xai_map_mask = cv2.resize(xai_map_normalized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

        # use alpha mask to blend the image with the colored xai_map
        img = img * (1 - xai_map_mask[..., None]) + xai_map_colored * xai_map_mask[..., None]
        img = img / np.max(img)
        return img

    def generate_video(self):
        ''' Generates video with the selected options from the menu. '''
        if self.video_length.get() > (len(self.ObjectDetector.dataset) - self.data_idx):
            self.show_message(f"Video lenght should be between 2 and {len(self.ObjectDetector.dataset) - self.data_idx}") 
            return False
        
        self.video_folder = f"videos/video"
        if hasattr(self, "video_scene"):
            self.video_folder += f"_scene{self.video_scene}"
        else:
            self.video_folder += f"_{self.data_idx}_{self.video_length.get()}"

        if os.path.isdir(self.video_folder):
            shutil.rmtree(self.video_folder)
        os.makedirs(self.video_folder)

        self.img_labels = []
        self.start_video_idx = self.data_idx
        self.video_gen_bool = True
        print(f"\nGenerating video frames inside \"{self.video_folder}\"...")
        prog_bar = mmcv.ProgressBar(self.video_length.get())

        # Loop through the dataset for generating the frame
        for i in range(self.data_idx, self.data_idx + self.video_length.get()):
            self.data_idx = i
            labels = self.generate_video_frame()
            self.img_labels.append(labels)
            prog_bar.update()

        data = {"img_labels": self.img_labels}
        file_path = os.path.join(self.video_folder, "labels.pkl")

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        self.video_gen_bool = False

        self.show_message(f"Video generated inside \"{self.video_folder}\" folder")
    
    def generate_video_frame(self):
        ''' Generates a single frame composed of 6 (cameras) saliency maps. '''

        # Extract camera images
        self.update_data(initialize_bboxes=True)

        # Check if only one object is selected
        if self.single_bbox.get():
            self.update_objects_list(single_select=True, all_select=False)
            self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]
        else:
            self.bbox_idx = list(range(len(self.labels)))

        # Generates XAI maps
        if self.overlay_bool.get():
            if self.selected_expl_type.get() in ["Grad-CAM", "Gradient Rollout"]:
                self.update_data(gradients=True, initialize_bboxes=False)

            if not self.no_object:
                self.ExplainableModel.generate_explainability(
                    self.selected_expl_type.get(),
                    self.selected_head_fusion.get(),
                    self.handle_residual.get(),
                    self.apply_rule.get())

                self.ExplainableModel.select_explainability(
                    self.nms_idxs, self.bbox_idx,
                    self.selected_discard_threshold.get(),
                    self.selected_map_quality.get(),
                    remove_pad=self.remove_pad.get(),
                    layer_fusion_method=self.selected_layer_fusion_type.get())

                if self.single_bbox.get():
                    cam_obj = self.get_object_camera()
                    if cam_obj == -1:
                        self.show_message("Please change the selected options")
                        return
                    self.selected_camera = cam_obj
        
        # Generate saliency maps for the 6 images
        cam_imgs = []  
        for camidx in range(len(self.imgs)):
            
            if self.draw_bboxes.get():
                img, _ = draw_lidar_bbox3d_on_img(
                        self.pred_bboxes,
                        self.imgs[camidx],
                        self.img_metas['lidar2img'][camidx],
                        color=(0, 255, 0),
                        with_bbox_id=True,
                        all_bbx=True,
                        bbx_idx=self.bbox_idx,
                        mode_2d=self.bbox_2d.get(),
                        labels=None if self.single_bbox.get() else self.labels)
            else:
                img = self.imgs[camidx]
            
            cam_layers = []
            if not self.no_object and self.overlay_bool.get():
                if self.aggregate_layers.get():
                    xai_map = self.ExplainableModel.xai_maps[camidx]
                    saliency_map = self.generate_saliency_map(img, xai_map)        
                    saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                    cam_layers.append(saliency_map)
                else:
                    for layer in range(len(self.ExplainableModel.xai_layer_maps)):
                        xai_map = self.ExplainableModel.xai_layer_maps[layer][camidx]
                        saliency_map = self.generate_saliency_map(img, xai_map)  
                        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                        cam_layers.append(saliency_map)
                
                cam_imgs.append(cam_layers)   
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cam_layers.append(img)
                cam_imgs.append(cam_layers)  

        # Generate a single image by concatenating the 6 images and save it as a frame.
        for layer in range(len(cam_imgs[0])):
            hori = np.concatenate((cam_imgs[2][layer], cam_imgs[0][layer], cam_imgs[1][layer]), axis=1)
            ver = np.concatenate((cam_imgs[5][layer], cam_imgs[3][layer], cam_imgs[4][layer]), axis=1)
            img = np.concatenate((hori, ver), axis=0)
            if not self.no_object and self.overlay_bool.get():
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            name = self.selected_expl_type.get().replace(" ", "_")
            data_idx_str = str(self.data_idx).zfill(4) 
            file_name = f"{name}_{data_idx_str}.jpeg"
            layer_folder = os.path.join(self.video_folder, f"layer_{layer}")
            if not os.path.exists(layer_folder):
                os.makedirs(layer_folder)
            file_path = os.path.join(layer_folder, file_name)
            img.save(file_path)

        return self.labels

    def load_video(self):
        ''' MLoads a selected video folder and its labels and scene information. '''
        self.video_folder = fd.askdirectory(
                title='Load video directory',
                initialdir='/workspace/videos/')
                
        if not self.video_folder:
            print("No directory selected.")
            return
        labels_file = os.path.join(self.video_folder, "labels.pkl")
        self.img_labels = None

        target_classes = np.arange(0, 10, 1)
        if os.path.exists(labels_file):
            with open(labels_file, 'rb') as f:
                data = pickle.load(f)
            self.img_labels = data["img_labels"]
            self.target_classes = [[i for i, tensor in enumerate(self.img_labels) if target_class in tensor.tolist()] for target_class in target_classes]
        else:
            self.update_objects_list(labels=[])

        items = os.listdir(self.video_folder)
        if any(os.path.isdir(os.path.join(self.video_folder, item)) for item in items):
            layer_folders = [f for f in items if f.startswith('layer_') and os.path.isdir(os.path.join(self.video_folder, f))]
            layer_folders.sort(key=lambda x: int(x.split('_')[-1]))  # Sort the folders by the layer number
        else:  
            layer_folders = ['']

        self.img_frames = []
        for folder in layer_folders:
            folder_path = os.path.join(self.video_folder, folder)
            folder_images = os.listdir(folder_path)
            folder_images.sort(key=lambda x: int(x.split('_')[-1].split(".")[0]))
            images = [Image.open(os.path.join(folder_path, img)) for img in folder_images]
            self.img_frames.append(images)

        # Assuming that there's at least one image
        if len(self.img_frames) > 0:
            self.start_video_idx = int(folder_images[0].split('.')[0].split('_')[-1])
            self.video_length.set(len(self.img_frames[0]))
            self.layers_video = len(self.img_frames)
            self.target_classes.append(list(range(self.video_length.get())))
        else:
            self.show_message("The folder should contain at least one image or a folder with images!")
            return

        # Extract video scene description from folder name
        parts = self.video_folder.split('_')
        scene_index = None
        for part in parts:
            if 'scene' in part:
                scene_index_str = part.replace('scene', '')
                if scene_index_str.isdigit():
                    scene_index = int(scene_index_str)
                    break 

        # Extract video scene descriptions from start index and video_lenght
        if scene_index is None:
            self.scene_indices = {}
            sum_samples = 0
            for i, samples in enumerate(self.scene_samples):
                sum_samples += samples
                if sum_samples >= self.start_video_idx:
                    self.scene_indices[sum_samples] = i 
                if sum_samples >= self.video_length.get():
                    break
            scene_index = list(self.scene_indices.values())[0] 

        # Update scene description accordingly
        self.scene_description = self.scene_descriptions[scene_index]
        self.update_info_video_label()

        if hasattr(self, "scale"):
            self.scale.configure(to=self.video_length.get())

        self.show_message(f"Video loaded ({self.video_length.get()} images).")

        if not self.video_loaded:
            self.menubar.add_command(label="Show video", command=self.show_video)
            self.add_separator("|")
            self.video_loaded = True

    def update_object_filter(self):
        ''' Manages the visualization of the frames containing a specific object class. '''
        self.target_class = self.target_classes[self.selected_filter.get()]
        if len(self.target_class) == 0:
            self.show_message(f"No {self.ObjectDetector.class_names[self.selected_filter.get()]} found")
            self.target_class = self.target_classes[-1]
        self.video_length.set(len(self.target_class))
        if hasattr(self, "scale"):
            self.scale.configure(to=self.video_length.get())
            self.idx_video.set(max(0, min(self.video_length.get(), self.data_idx - self.start_video_idx)))
        else:
            self.idx_video = tk.IntVar()
            self.scale = tk.Scale(self.frame, from_=0, to=self.video_length.get(), variable=self.idx_video, command=self.update_index, showvalue=False, orient='horizontal')
        self.scale.pack(fill='x')

    def update_index(self, event=None):
        ''' Updates the video index if arrow keys are pressed and shows the frame. '''
        if self.paused:
            if not isinstance(event, str):
                if self.flag: 
                    return
                self.flag = True
                self.after(self.delay, lambda: setattr(self, 'flag', False))  # Reset flag after delay
                if event.keysym == 'Right':
                    self.idx_video.set(self.idx_video.get() + 1)
                elif event.keysym == 'Left':
                    self.idx_video.set(self.idx_video.get() - 1)
                elif event.keysym == 'Up':
                    self.layer_idx = max(0, self.layer_idx - 1)
                elif event.keysym == 'Down':
                    self.layer_idx = min(self.layers_video-1, self.layer_idx + 1)
                self.frame.focus_set()
            self.show_sequence(forced=True)
            if hasattr(self, "img_labels"):
                labels = self.img_labels[self.target_class[self.idx_video.get()]]
                self.update_objects_list(labels=labels, single_select=True)

    def pause_resume(self, event=None):
        ''' Manages the pausing/resuming of the video.'''
        if not self.paused:
            self.after_cancel(self.after_seq_id)
            self.paused = True
            self.idx_video.set(self.idx_video.get() - 1)
            if hasattr(self, "img_labels"):
                labels = self.img_labels[self.target_class[self.idx_video.get()]]
                self.update_objects_list(labels=labels, single_select=False)
        else:
            self.paused = False
            self.show_sequence()
        self.frame.focus_set()

    def show_sequence(self, forced=False):
        ''' Shows the loaded video frame by frame. '''
        if not self.paused or forced:
            if self.idx_video.get() >= self.video_length.get():
                self.idx_video.set(0)

            # Extracts the image frame and adapt the Tkinter canvas for it
            img_frame = self.img_frames[self.layer_idx][self.target_class[self.idx_video.get()]]
            self.w, self.h = self.canvas.winfo_width(), self.canvas.winfo_height()

            # If window is resized, update the canvas
            if (self.old_w, self.old_h) != (self.w, self.h):
                img_w, img_h = img_frame.width, img_frame.height
                canvas_ratio, img_ratio = self.w / self.h, img_w / img_h

                if img_ratio > canvas_ratio:
                    self.new_w, self.new_h = self.w, int(self.w / img_ratio)
                else:
                    self.new_w, self.new_h = int(self.h * img_ratio), self.h

                x = (self.w - self.new_w) // 2
                y = (self.h - self.new_h) // 2
                self.canvas.coords("img_tag", x, y)

                self.old_w, self.old_h = self.w, self.h

            self.img_frame = ImageTk.PhotoImage(img_frame.resize((self.new_w, self.new_h)))
            self.canvas.itemconfig(self.canvas_frame, image=self.img_frame)
            
            self.data_idx = self.start_video_idx + self.target_class[self.idx_video.get()]
            self.update_info_label()
            if hasattr(self, "scene_indices"):
                if self.data_idx in self.scene_indices:
                    scene_index = self.scene_indices[self.data_idx] + 1
                    if self.scene_description != self.scene_descriptions[scene_index]:
                        self.scene_description = self.scene_descriptions[scene_index]
                    else:
                        self.scene_description = self.scene_descriptions[scene_index-1]
                    self.update_info_video_label()

            if not forced:
                self.idx_video.set(self.idx_video.get() + 1)
                self.after_seq_id = self.after(self.video_delay.get(), self.show_sequence)


