import tkinter as tk
import numpy as np
import torch
import cv2
import mmcv
import os
import random
import pickle
import tomli
from tkinter import filedialog as fd
from mmcv.parallel import DataContainer as DC
from tkinter.messagebox import showinfo
from tkinter import scrolledtext
import matplotlib.pyplot as plt
from PIL import Image, ImageTk


from modules.Configs import Configs
from modules.Explainability import ExplainableTransformer
from modules.Explainability  import generate_saliency_map
from modules.Model import Model
from scripts.evaluate import evaluate


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
        self.tk.call("set_theme", "dark")
        self.title('Explainable Transformer-based 3D Object Detector')
        self.geometry('1500x1500')
        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.canvas, self.fig, self.spec, self.single_object_window, self.single_object_canvas = None, None, None, None, None

        # Model object
        self.ObjectDetector = Model()
        self.indices_file = 'misc/indices.txt'
        if not os.path.exists(self.indices_file):
            open(self.indices_file, 'w').close()
        self.file_suffix = 0
        self.bg_color = self.option_get('background', '*')
        self.started_app = False
        self.video_gen_bool = False

        self.old_layer = None
        self.old_bbox_idx = None

        self.bbox_coords, self.saliency_maps_objects = [], []
        
        # Main Tkinter menu in which all other cascade menus are added
        self.menubar = tk.Menu(self)

        # Cascade menus for loading model and selecting the GPU
        self.config(menu=self.menubar)
        file_opt, gpu_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.gpu_id = tk.IntVar()
        self.gpu_id.set(0)
        file_opt.add_command(label=" Load model", command=self.load_model)
        file_opt.add_command(label=" Load model from config file", command=lambda: self.load_model(from_config=True))
        file_opt.add_command(label=" Save index", command=lambda: self.insert_entry(type=1))
        file_opt.add_command(label=" Capture screen", command=self.capture)
        file_opt.add_cascade(label=" Gpu", menu=gpu_opt)
        file_opt.add_separator()
        file_opt.add_command(label=" Show car setup", command=self.show_car)
        file_opt.add_command(label=" How to use", command=self.show_app_info)
        
        message = "You need to reload the model to apply GPU change."
        for i in range(torch.cuda.device_count()):
            gpu_opt.add_radiobutton(label=f"GPU {i}", variable=self.gpu_id, value=i, command=lambda: self.show_message(message))

        self.menubar.add_cascade(label=" File", menu=file_opt)

    def load_model(self, from_config=False):
        if not from_config:
            self.ObjectDetector.load_model(gpu_id=self.gpu_id.get())
        else:
            self.ObjectDetector.load_from_config()

        self.ExplainableModel = ExplainableTransformer(self.ObjectDetector)

        # Synced configurations: when a value is changed, the triggered function is called
        data_configs, expl_configs = [], []
        self.data_configs = Configs(data_configs, triggered_function=self.update_data, type=0)
        self.expl_configs = Configs(expl_configs, triggered_function=self.ExplainableModel.generate_explainability, type=1)

        if not self.started_app:
            print("Starting app...\n")
            self.start_app()
            self.random_data_idx()
            self.started_app = True
            print("Completed.\n")

        self.update_info_label()

    def start_app(self):
        '''
        It starts the UI after loading the model. Variables are initialized.
        '''
        self.frame = tk.Frame(self)
        self.frame.pack(fill=tk.Y)
        self.info_text = tk.StringVar()
        self.info_label = tk.Label(self.frame, textvariable=self.info_text, anchor=tk.CENTER)
        self.info_label.bind("<Button-1>", lambda event: self.show_model_info())
        self.info_label.bind("<Enter>", lambda event: self.red_text())
        self.info_label.bind("<Leave>", lambda event: self.black_text())
        self.info_label.pack(side=tk.TOP)

        # Cascade menu for Data settings
        dataidx_opt, self.select_idx_opt, saved_indices, thr_opt = tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.selected_threshold = tk.DoubleVar()
        self.selected_threshold.set(0.5)
        values = np.arange(0.0, 1, 0.1).round(1)
        for i in values:
            thr_opt.add_radiobutton(label=i, variable=self.selected_threshold)


        self.select_idx_opt.add_command(label="Insert index", command=lambda: self.insert_entry(type=0))
        self.selected_data_idx = tk.IntVar()
        with open(self.indices_file, 'r') as file:
            for line in file:
                self.select_idx_opt.add_radiobutton(label=line.strip(), variable=self.selected_data_idx, command=self.update_idx, value=line.split()[0])
        dataidx_opt.add_cascade(label=" Select data index", menu=self.select_idx_opt)
        dataidx_opt.add_command(label=" Select random data", command=self.random_data_idx)
        dataidx_opt.add_cascade(label=" Select prediction threshold", menu=thr_opt)
        dataidx_opt.add_separator()
        dataidx_opt.add_command(label=" Show LiDAR", command=self.show_lidar)


        framerate_opt = tk.Menu(self.menubar)
        frame_rates = np.arange(0, 35, 5)
        frame_rates[0] = 1
        self.frame_rate = tk.IntVar()
        self.frame_rate.set(frame_rates[0])
        for i in range(len(frame_rates)):
            framerate_opt.add_radiobutton(label=frame_rates[i], variable=self.frame_rate, value=frame_rates[i])

        videolength_opt = tk.Menu(self.menubar)
        video_lengths = np.arange(10, 200, 20)
        self.video_length = tk.IntVar()
        self.video_length.set(video_lengths[0])
        for i in range(len(video_lengths)):
            videolength_opt.add_radiobutton(label=video_lengths[i], variable=self.video_length , value=video_lengths[i])

        video_opt = tk.Menu(self.menubar)
        video_opt.add_command(label=" Generate", command=self.generate_video)
        video_opt.add_command(label=" Load", command=self.load_video)
        video_opt.add_command(label=" Save", command=self.save_video)
        video_opt.add_command(label=" Show", command=self.show_video)
        video_opt.add_cascade(label=" Sequence length", menu=videolength_opt)
        video_opt.add_cascade(label=" Frame rate", menu=framerate_opt)

        # Cascade menu for Camera
        self.cameras = {'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5}
        self.cam_idx = [2, 0, 1, 5, 3, 4]  # Used for visualizing camera outputs properly

        # Cascade menu for Attention layer
        layer_opt = tk.Menu(self.menubar)
        self.selected_layer = tk.IntVar()
        for i in range(self.ObjectDetector.num_layers):
            layer_opt.add_radiobutton(label=i, variable=self.selected_layer)
        self.selected_layer.set(self.ExplainableModel.num_layers - 1)

        # Cascade menus for Explainable options
        expl_opt = tk.Menu(self.menubar)
        raw_attention, grad_cam, grad_rollout = tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.expl_options = ["Raw Attention", "Grad-CAM", "Gradient Rollout"]

        # Raw Attention
        expl_opt.add_cascade(label=self.expl_options[0], menu=raw_attention)
        self.head_fusion_types = ["max", "min", "mean"]
        self.selected_head_fusion = tk.StringVar()
        self.selected_head_fusion.set(self.head_fusion_types[0])

        hf_opt = tk.Menu(self.menubar)
        self.selected_discard_threshold = tk.DoubleVar()
        self.selected_discard_threshold.set(0.1)
        values = np.arange(0.0, 1, 0.1).round(1)
        for i in range(len(self.head_fusion_types)):
            hf_opt.add_radiobutton(label=self.head_fusion_types[i].capitalize(), variable=self.selected_head_fusion, value=self.head_fusion_types[i])
        for head in range(self.ObjectDetector.num_heads):
            hf_opt.add_radiobutton(label=str(head), variable=self.selected_head_fusion, value = str(head))
        raw_attention.add_cascade(label=" Head", menu=hf_opt)

        # Grad-CAM
        expl_opt.add_cascade(label=self.expl_options[1], menu=grad_cam)
        self.grad_cam_types = ["default"]
        self.selected_gradcam_type = tk.StringVar()
        self.selected_gradcam_type.set(self.grad_cam_types[0])
        for i in range(len(self.grad_cam_types)):
            grad_cam.add_radiobutton(label=self.grad_cam_types[i].capitalize(), variable=self.selected_gradcam_type, value=self.grad_cam_types[i])

        # Gradient Rollout
        expl_opt.add_cascade(label=self.expl_options[2], menu=grad_rollout)
        self.handle_residual, self.apply_rule = tk.BooleanVar(), tk.BooleanVar()
        self.handle_residual.set(True)
        self.apply_rule.set(True)
        grad_rollout.add_checkbutton(label=" Handle residual", variable=self.handle_residual, onvalue=1, offvalue=0)
        grad_rollout.add_checkbutton(label=" Apply rule 10", variable=self.apply_rule, onvalue=1, offvalue=0)


        expl_opt.add_separator()

        # Explainable mechanism selection
        expl_type_opt = tk.Menu(self.menubar)
        expl_opt.add_cascade(label="Mechanism", menu=expl_type_opt)
        self.selected_expl_type = tk.StringVar()
        self.selected_expl_type.set(self.expl_options[0])
        for i in range(len(self.expl_options)):
            expl_type_opt.add_radiobutton(label=self.expl_options[i], variable=self.selected_expl_type, value=self.expl_options[i], command=self.update_info_label)
        
        pert_opt = tk.Menu(self.menubar)
        self.selected_pert_step = tk.DoubleVar()
        pert_steps = np.arange(0, 1, 0.1)
        for step in pert_steps:
            pert_opt.add_radiobutton(label=f"{int(step*100)} %", variable=self.selected_pert_step, value=step)
        expl_opt.add_cascade(label="Perturbate image", menu=pert_opt)

        # Discard ratio for attention weights
        dr_opt = tk.Menu(self.menubar)
        self.show_self_attention, self.gen_segmentation = tk.BooleanVar(), tk.BooleanVar()
        self.show_self_attention.set(True)
        for i in values:
            dr_opt.add_radiobutton(label=i, variable=self.selected_discard_threshold)
        expl_opt.add_cascade(label="Discard threshold", menu=dr_opt)
        expl_opt.add_checkbutton(label="Generate segmentation map", onvalue=1, offvalue=0, variable=self.gen_segmentation)


        # Cascade menus for object selection
        self.bbox_opt = tk.Menu(self.menubar)
        self.single_bbox = tk.BooleanVar()
        self.select_all_bboxes = tk.BooleanVar()
        self.select_all_bboxes.set(True)
        self.bbox_opt.add_checkbutton(label=" Single object", onvalue=1, offvalue=0, variable=self.single_bbox, command=self.single_bbox_select) 
        self.bbox_opt.add_checkbutton(label=" Select all", onvalue=1, offvalue=0, variable=self.select_all_bboxes, command=self.initialize_bboxes)
        self.bbox_opt.add_separator()

        quality_opt = tk.Menu(self.menubar)
        map_qualities = ["Low", "Medium", "High"]
        self.selected_map_quality = tk.StringVar()
        self.selected_map_quality.set(map_qualities[2])
        for i in range(len(map_qualities)):
            quality_opt.add_radiobutton(label=map_qualities[i], variable=self.selected_map_quality, value=map_qualities[i])

        # Cascade menus for Additional options
        add_opt = tk.Menu(self.menubar)
        self.GT_bool, self.overlay_bool, self.bbox_2d, self.show_self_attention = \
            tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.overlay_bool.set(True)
        self.bbox_2d.set(True)
        self.show_self_attention.set(True)
        add_opt.add_checkbutton(label=" 2D bounding boxes", onvalue=1, offvalue=0, variable=self.bbox_2d)
        add_opt.add_checkbutton(label=" Show GT Bounding Boxes", onvalue=1, offvalue=0, variable=self.GT_bool)
        add_opt.add_checkbutton(label=" Saliency maps on images", onvalue=1, offvalue=0, variable=self.overlay_bool)
        add_opt.add_checkbutton(label=" Show objects self-attention", onvalue=1, offvalue=0, variable=self.show_self_attention)
        add_opt.add_cascade(label=" Select maps quality", menu=quality_opt)
        add_opt.add_command(label=" Change theme", command=self.change_theme)

        # Adding all cascade menus ro the main menubar menu
        self.add_separator()
        self.menubar.add_cascade(label="Video", menu=video_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Data", menu=dataidx_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Objects", menu=self.bbox_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Layer", menu=layer_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Explainability", menu=expl_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Settings", menu=add_opt)
        self.add_separator("|")
        self.menubar.add_command(label="Visualize", command=self.visualize)
        self.add_separator("|")


    def show_car(self):
        img = plt.imread("misc/car.png")
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        
    def update_idx(self):
        self.data_idx = self.selected_data_idx.get()
        self.update_info_label()

    def show_app_info(self):
        readme_path = 'README.md'
        with open(readme_path, 'r') as f:
            readme_content = f.read()

        readme_window = tk.Toplevel(self)
        readme_window.title("App information")

        readme_text = tk.Text(readme_window, wrap=tk.WORD)
        readme_text.insert(tk.END, readme_content)
        readme_text.configure(state='disabled')
        readme_text.pack()

    def update_data(self, select_layer=True, initialize_bboxes=True, pert_step=None):
        '''
        Predict bboxes and extracts attentions.
        '''
        # Load selected data from dataloader, manual DataContainer fixes are needed
        data = self.ObjectDetector.dataset[self.data_idx]
        metas = [[data['img_metas'][0].data]]
        img = [data['img'][0].data.unsqueeze(0)]
        data['img_metas'][0] = DC(metas, cpu_only=True)
        data['img'][0] = DC(img)
        self.data = data

        img_norm_cfg = self.ObjectDetector.cfg.get('img_norm_cfg')
        mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
        std = np.array(img_norm_cfg["std"], dtype=np.float32)

        if "points" in self.data.keys():
            self.data.pop("points")

        if not pert_step:
            # Attention scores are extracted, together with gradients if grad-CAM is selected
            if self.selected_expl_type.get() not in ["Grad-CAM", "Gradient Rollout"]:
                outputs = self.ExplainableModel.extract_attentions(self.data)
            else:
                outputs = self.ExplainableModel.extract_attentions(self.data, self.bbox_idx)
        else:
            xai_maps = self.ExplainableModel.xai_maps[self.selected_layer.get()]
            img = img[0][0]
            img = img[:, :, :self.ObjectDetector.ori_shape[0], :self.ObjectDetector.ori_shape[1]]  # [num_cams x height x width x channels]

            mask = torch.Tensor(-mean)
            img_pert_list = []
            for camidx in range(len(xai_maps)):
                #  {'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5}

                img_pert = img[camidx].permute(1, 2, 0).numpy()
                xai = xai_maps[camidx]
                filter_mask = xai > 0.2

                # apply the filter_mask to xai and flatten it
                filtered_xai = xai[filter_mask].flatten()

                # create a tensor of original indices
                original_indices = torch.arange(xai.numel()).reshape(xai.shape)[filter_mask].flatten()

                # compute the threshold for top 20% values
                top_k = int(pert_step * filtered_xai.numel())

                # get the top k values and their indices
                values, indices = torch.topk(filtered_xai, top_k)

                # get the indices with respect to the original tensor
                original_indices = original_indices[indices]

                # convert the 1D indices back to 2D
                row_indices, col_indices = original_indices // xai.size(1), original_indices % xai.size(1)

                img_pert[row_indices, col_indices] = mask

                img_pert_list.append(img_pert)
            if len(img_pert_list) > 0:
                # save_img the perturbed 6 camera images into the data input
                img_pert_list = torch.from_numpy(np.stack(img_pert_list))
                img = [img_pert_list.permute(0, 3, 1, 2).unsqueeze(0)] # img = [torch.Size([1, 6, 3, 928, 1600])
                self.data['img'][0] = DC(img)
                
                with torch.no_grad():
                    outputs = self.ObjectDetector.model(return_loss=False, rescale=True, all_layers=True, **self.data)


        # Those are needed to index the bboxes decoded by the NMS-Free decoder
        self.nms_îdxs_layers = self.ObjectDetector.model.module.pts_bbox_head.bbox_coder.get_indexes()
        self.outputs = outputs[0]["pts_bbox"]
        self.thr_idxs_layers = [output_layer['scores_3d'] > self.selected_threshold.get() for output_layer in self.outputs]
        self.pred_bboxes_layers = [output_layer["boxes_3d"][thr_layer] for output_layer,thr_layer in zip(self.outputs, self.thr_idxs_layers)]
        for i in range(len(self.pred_bboxes_layers)):
            self.pred_bboxes_layers[i].tensor.detach()
                       
        self.labels_layers = [output_layer['labels_3d'][thr_layer] for output_layer,thr_layer in zip(self.outputs, self.thr_idxs_layers)]

        self.no_object = False
        if len(self.labels_layers[self.selected_layer.get()]) == 0:
            self.no_object = True
            if not self.video_gen_bool:
                print("No object detected.")
                self.show_message("No object detected.")

        # Extract image metas which contain, for example, the lidar to camera projection matrices
        self.img_metas = self.data["img_metas"][0]._data[0][0]

        # Extract the 6 camera images from the data and remove the padded pixels
        imgs = self.data["img"][0]._data[0].numpy()[0]
        # imgs = imgs.transpose(0, 2, 3, 1)
        imgs = imgs.transpose(0, 2, 3, 1)[:, :self.ObjectDetector.ori_shape[0], :self.ObjectDetector.ori_shape[1], :]  # [num_cams x height x width x channels]
        
        # Denormalize the images
        for i in range(len(imgs)):
            imgs[i] = mmcv.imdenormalize(imgs[i], mean, std, to_bgr=False)
        self.imgs = imgs.astype(np.uint8)

        if select_layer:
            if pert_step:
                self.select_layer(initialize_bboxes=initialize_bboxes, all_select=False)
            else:
                self.select_layer(initialize_bboxes=initialize_bboxes)

    
    def select_layer(self, all_select=True, initialize_bboxes=True):
        # Extract only the selected layer
        self.labels = self.labels_layers[self.selected_layer.get()]
        self.nms_idxs = self.nms_îdxs_layers[self.selected_layer.get()]
        self.thr_idxs = self.thr_idxs_layers[self.selected_layer.get()]
        self.pred_bboxes = self.pred_bboxes_layers[self.selected_layer.get()]

        if initialize_bboxes and not self.video_gen_bool:
            self.update_objects_list()
            self.initialize_bboxes(all_select=all_select)

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
        # Type 0: data index; type 1: video lenght; type 2: frame rate
        popup = tk.Toplevel(self)
        popup.geometry("80x50")

        self.entry = tk.Entry(popup, width=20)
        self.entry.pack()

        button = tk.Button(popup, text="OK", command=lambda: self.close_entry(popup, type))
        button.pack()

    def close_entry(self, popup, type):
        entry = self.entry.get()
        if type == 0:
            if entry.isnumeric() and int(entry) <= (len(self.ObjectDetector.dataset)-1):
                self.data_idx = int(entry)
                self.update_info_label()
            else:
                self.show_message(f"Insert an integer between 0 and {len(self.ObjectDetector.dataset)}")
        elif type == 1:
            self.select_idx_opt.add_radiobutton(label=f'{self.data_idx} | {entry}', variable=self.selected_data_idx, command=self.update_idx, value=self.data_idx)
            with open(self.indices_file, 'a') as file:
                file.write(f'{self.data_idx} | {entry}\n')
        
        popup.destroy()
        
    def random_data_idx(self):
        idx = random.randint(0, len(self.ObjectDetector.dataset)-1)
        self.data_idx = idx
        self.update_info_label()

    def update_info_label(self, info=None, idx=None):
        if idx is None:
            idx = self.data_idx
        if info is None:
            info = f"Model: {self.ObjectDetector.model_name} | Dataloader: {self.ObjectDetector.dataloader_name} | Data index: {idx} | Mechanism: {self.selected_expl_type.get()}"
        
        self.info_text.set(info)

    def update_objects_list(self, labels=None):
        if labels is None:
            labels = self.labels
        self.bboxes = []
        self.bbox_opt.delete(3, 'end')
        for i in range(len(labels)):
            view_bbox = tk.BooleanVar()
            view_bbox.set(False)
            self.bboxes.append(view_bbox)
            self.bbox_opt.add_checkbutton(label=f" {self.ObjectDetector.class_names[labels[i].item()].capitalize()} ({i})", onvalue=1, offvalue=0, variable=self.bboxes[i], command=lambda idx=i: self.single_bbox_select(idx))

    def single_bbox_select(self, idx=None, single_select=False):
        self.select_all_bboxes.set(False)
        if single_select:
            self.single_bbox.set(True)
        if self.single_bbox.get():
            if idx is None:
                idx = 0
            for i in range(len(self.bboxes)):
                if i != idx:
                    self.bboxes[i].set(False)

    def initialize_bboxes(self, all_select=False):
        if all_select:
            self.select_all_bboxes.set(True)
        if hasattr(self, "bboxes"):
            if self.select_all_bboxes.get():
                self.single_bbox.set(False)
                for i in range(len(self.bboxes)):
                    self.bboxes[i].set(True)
            else:
                if len(self.bboxes) > 0:
                    self.bboxes[0].set(True)

    def get_camera_object(self):
        scores = self.ExplainableModel.scores
        if not scores:
            return -1
        cam_obj = scores[self.selected_layer.get()].index(max(scores[self.selected_layer.get()]))
        return cam_obj

    def capture(self):
        screenshots_path = "screenshots/"
        if not os.path.exists(screenshots_path):
            os.makedirs(screenshots_path)

        path = screenshots_path + f"{self.ObjectDetector.model_name}_{self.data_idx}"

        if os.path.exists(path+"_"+str(self.file_suffix)+".png"):
            self.file_suffix += 1
        else:
            self.file_suffix = 0

        path += "_" + str(self.file_suffix) + ".png"
        self.fig.savefig(path, dpi=300, transparent=True)
        print(f"Screenshot saved in {path}.")

    def change_theme(self):
        if self.tk.call("ttk::style", "theme", "use") == "azure-dark":
            self.tk.call("set_theme", "light")
            self.bg_color = "white"
        else:
            self.tk.call("set_theme", "dark")
            self.bg_color = self.option_get('background', '*')

        self.fig.set_facecolor(self.bg_color)
        self.canvas.draw()

    def save_video(self):
        if hasattr(self, "img_frames"):
            data = {'img_frames': self.img_frames, "img_labels": self.img_labels, "video_idx": self.start_video_idx, "video_lenght": self.video_length.get()}

            file_path = fd.asksaveasfilename(defaultextension=".pkl", filetypes=[("All Files", "*.*")])

            print(f"Saving video in {file_path}...\n")
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Video saved.")
        else:
            self.show_message("You should first generate a video.")

    def load_video(self):
        video_datatypes = (
            ('Pickle', '*.pkl'),
        )
        video_pickle = fd.askopenfilename(
            title='Load video data',
            initialdir='/workspace/',
            filetypes=video_datatypes)

        print(f"Loading video from {video_pickle}...\n")
        with open(video_pickle, 'rb') as f:
            data = pickle.load(f)

        self.img_frames = data["img_frames"]
        self.img_labels = data["img_labels"]
        self.start_video_idx = data["video_idx"]
        self.video_length.set(data["video_lenght"])

        if hasattr(self, "scale"):
            self.scale.configure(to=self.video_length.get())

        print(f"Video loaded from {video_pickle}.\n")