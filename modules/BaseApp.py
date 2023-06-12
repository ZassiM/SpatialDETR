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
from PIL import ImageGrab
import matplotlib.pyplot as plt

from modules.Configs import Configs
from modules.Explainability import ExplainableTransformer
from modules.Explainability  import overlay_attention_on_image
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

        self.started_app = False
        self.video_length = 15
        self.video_gen_bool = False
        self.frame_rate = 1
        
        # Main Tkinter menu in which all other cascade menus are added
        self.menubar = tk.Menu(self)

        # Cascade menus for loading model and selecting the GPU
        self.config(menu=self.menubar)
        file_opt, gpu_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.gpu_id = tk.IntVar()
        self.gpu_id.set(0)
        file_opt.add_command(label=" Load model", command=self.load_model)
        file_opt.add_command(label=" Load model from config file", command=lambda: self.load_model(from_config=True))
        file_opt.add_command(label=" Load video from pickle file", command=self.load_video)
        file_opt.add_command(label=" Save video", command=self.save_video)
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
        # Suffix used for saving screenshot of same model with different numbering
        self.file_suffix = 0

        # Synced configurations: when a value is changed, the triggered function is called
        data_configs, expl_configs = [], []
        self.data_configs = Configs(data_configs, triggered_function=self.update_data, type=0)
        self.expl_configs = Configs(expl_configs, triggered_function=self.ExplainableModel.generate_explainability, type=1)

        # Tkinter frame for visualizing model and GPU info
        frame = tk.Frame(self)
        frame.pack(fill=tk.Y)
        self.info_text = tk.StringVar()
        self.info_label = tk.Label(frame, textvariable=self.info_text, anchor=tk.CENTER)
        self.info_label.bind("<Button-1>", lambda event: self.show_model_info())
        self.info_label.bind("<Enter>", lambda event: self.red_text())
        self.info_label.bind("<Leave>", lambda event: self.black_text())
        self.info_label.pack(side=tk.TOP)
        self.bg_color = self.info_label.cget("background")

        # Cascade menu for Data index
        dataidx_opt = tk.Menu(self.menubar)
        dataidx_opt.add_command(label=" Select data index", command=lambda: self.select_data_idx(type=0))
        dataidx_opt.add_command(label=" Select random data", command=self.random_data_idx)
        dataidx_opt.add_separator()

        dataidx_opt.add_command(label=" Select video length", command=lambda: self.select_data_idx(type=1))
        dataidx_opt.add_command(label=" Select frame rate", command=lambda: self.select_data_idx(type=2))
        dataidx_opt.add_command(label=" Generate video", command=self.generate_video)

        # Cascade menus for Prediction threshold
        thr_opt = tk.Menu(self.menubar)
        self.selected_threshold = tk.DoubleVar()
        self.selected_threshold.set(0.5)
        values = np.arange(0.0, 1, 0.1).round(1)
        for i in values:
            thr_opt.add_radiobutton(label=i, variable=self.selected_threshold, command=self.update_thr)

        # Cascade menu for Camera
        camera_opt = tk.Menu(self.menubar)
        self.cameras = {'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5}
        self.cam_idx = [2, 0, 1, 5, 3, 4]  # Used for visualizing camera outputs properly
        self.selected_camera = tk.IntVar()
        for value, key in enumerate(self.cameras):
            camera_opt.add_radiobutton(label=key, variable=self.selected_camera, value=value)
        camera_opt.add_radiobutton(label="All", variable=self.selected_camera, value=-1)
        self.selected_camera.set(-1) # Default: visualize all cameras

        # Cascade menu for Attention layer
        layer_opt = tk.Menu(self.menubar)
        self.selected_layer = tk.IntVar()
        self.show_all_layers = tk.BooleanVar()
        for i in range(self.ObjectDetector.num_layers):
            layer_opt.add_radiobutton(label=i, variable=self.selected_layer)
        layer_opt.add_checkbutton(label="All", onvalue=1, offvalue=0, variable=self.show_all_layers, command=self.check_layers)
        self.selected_layer.set(self.ExplainableModel.num_layers - 1)

        # Cascade menus for Explainable options
        expl_opt = tk.Menu(self.menubar)
        attn_rollout, grad_cam, grad_rollout = tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.expl_options = ["Attention Rollout", "Grad-CAM", "Gradient Rollout"]

        # Attention Rollout
        expl_opt.add_cascade(label=self.expl_options[0], menu=attn_rollout)
        self.head_fusion_types = ["max", "min", "mean"]
        self.selected_head_fusion = tk.StringVar()
        self.selected_head_fusion.set(self.head_fusion_types[0])
        self.raw_attn = tk.BooleanVar()
        self.raw_attn.set(True)
        hf_opt = tk.Menu(self.menubar)
        self.selected_discard_ratio = tk.DoubleVar()
        self.selected_discard_ratio.set(0.5)
        values = np.arange(0.0, 1, 0.1).round(1)
        for i in range(len(self.head_fusion_types)):
            hf_opt.add_radiobutton(label=self.head_fusion_types[i].capitalize(), variable=self.selected_head_fusion, value=self.head_fusion_types[i])
        for head in range(self.ObjectDetector.num_heads):
            hf_opt.add_radiobutton(label=str(head), variable=self.selected_head_fusion, value = str(head))
        attn_rollout.add_cascade(label=" Head fusion", menu=hf_opt)
        attn_rollout.add_checkbutton(label=" Raw attention", variable=self.raw_attn, onvalue=1, offvalue=0)

        # Grad-CAM
        expl_opt.add_cascade(label=self.expl_options[1], menu=grad_cam)
        self.grad_cam_types = ["default"]
        self.selected_gradcam_type = tk.StringVar()
        self.selected_gradcam_type.set(self.grad_cam_types[0])
        for i in range(len(self.grad_cam_types)):
            grad_cam.add_radiobutton(label=self.grad_cam_types[i].capitalize(), variable=self.selected_gradcam_type, value=self.grad_cam_types[i])

        # Gradient Rollout
        expl_opt.add_cascade(label=self.expl_options[2], menu=grad_rollout)
        self.handle_residual, self.apply_rule, self.apply_rollout = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.apply_rollout.set(True)
        self.handle_residual.set(True)
        self.apply_rule.set(True)
        grad_rollout.add_checkbutton(label=" Apply rollout", variable=self.apply_rollout, onvalue=1, offvalue=0)
        grad_rollout.add_checkbutton(label=" Handle residual", variable=self.handle_residual, onvalue=1, offvalue=0)
        grad_rollout.add_checkbutton(label=" Apply rule 10", variable=self.apply_rule, onvalue=1, offvalue=0)


        expl_opt.add_separator()

        # Explainable mechanism selection
        expl_type_opt = tk.Menu(self.menubar)
        expl_opt.add_cascade(label="Mechanism", menu=expl_type_opt)
        self.selected_expl_type = tk.StringVar()
        self.selected_expl_type.set(self.expl_options[0])
        self.old_expl_type = self.expl_options[0]
        for i in range(len(self.expl_options)):
            expl_type_opt.add_radiobutton(label=self.expl_options[i], variable=self.selected_expl_type, value=self.expl_options[i], command=self.update_info_label)

        # Discard ratio for attention weights
        dr_opt = tk.Menu(self.menubar)
        for i in values:
            dr_opt.add_radiobutton(label=i, variable=self.selected_discard_ratio)
        expl_opt.add_cascade(label="Discard ratio", menu=dr_opt)
        expl_opt.add_command(label="Evaluate explainability", command=lambda: evaluate(self.ObjectDetector, self.ExplainableModel, self.selected_expl_type.get()))
        # Cascade menus for object selection
        self.bbox_opt = tk.Menu(self.menubar)
        self.single_bbox = tk.BooleanVar()
        self.select_all_bboxes = tk.BooleanVar()
        self.select_all_bboxes.set(True)
        self.bbox_opt.add_checkbutton(label=" Single object", onvalue=1, offvalue=0, variable=self.single_bbox, command=self.single_bbox_select) 
        self.bbox_opt.add_checkbutton(label=" Select all", onvalue=1, offvalue=0, variable=self.select_all_bboxes, command=self.initialize_bboxes)
        self.bbox_opt.add_separator()

        # Cascade menus for Additional options
        add_opt = tk.Menu(self.menubar)
        self.show_attn, self.GT_bool, self.BB_bool, self.overlay_bool, self.show_labels, self.capture_bool, self.bbox_2d = \
            tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.show_attn.set(True)
        self.BB_bool.set(True)
        self.show_labels.set(True)
        self.overlay_bool.set(True)
        self.bbox_2d.set(True)
        add_opt.add_checkbutton(label=" Show attention maps", onvalue=1, offvalue=0, variable=self.show_attn, command=self.disable_attn)
        add_opt.add_checkbutton(label=" Show GT Bounding Boxes", onvalue=1, offvalue=0, variable=self.GT_bool)
        add_opt.add_checkbutton(label=" Show all Bounding Boxes", onvalue=1, offvalue=0, variable=self.BB_bool)
        add_opt.add_checkbutton(label=" Overlay attention on image", onvalue=1, offvalue=0, variable=self.overlay_bool)
        add_opt.add_checkbutton(label=" Show predicted labels", onvalue=1, offvalue=0, variable=self.show_labels)
        add_opt.add_checkbutton(label=" Capture output", onvalue=1, offvalue=0, variable=self.capture_bool)
        add_opt.add_checkbutton(label=" 2D bounding boxes", onvalue=1, offvalue=0, variable=self.bbox_2d)
        add_opt.add_command(label=" Change theme", command=self.change_theme)

        # Adding all cascade menus ro the main menubar menu
        self.add_separator()
        self.menubar.add_cascade(label="Data", menu=dataidx_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Prediction threshold", menu=thr_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Camera", menu=camera_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Objects", menu=self.bbox_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Layer", menu=layer_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Explainability", menu=expl_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Options", menu=add_opt)
        self.add_separator("|")
        self.menubar.add_command(label="Visualize", command=self.visualize)
        self.add_separator("|")
        self.menubar.add_command(label="Show LIDAR", command=self.show_lidar)
        self.add_separator("|")
        self.menubar.add_command(label="Show video", command=self.show_video)

    def show_car(self):
        img = plt.imread("misc/car.png")
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        
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

    def update_data(self, initialize_bboxes=True):
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

        if "points" in self.data.keys():
            self.data.pop("points")

        # Attention scores are extracted, together with gradients if grad-CAM is selected
        if self.selected_expl_type.get() not in ["Grad-CAM", "Gradient Rollout"]:
            outputs = self.ExplainableModel.extract_attentions(self.data)
        else:
            outputs = self.ExplainableModel.extract_attentions(self.data, self.bbox_idx)
        
        # Those are needed to index the bboxes decoded by the NMS-Free decoder
        self.nms_idxs = self.ObjectDetector.model.module.pts_bbox_head.bbox_coder.get_indexes()

        # Extract predicted bboxes and their labels
        self.outputs = outputs[0]["pts_bbox"]
        self.thr_idxs = self.outputs['scores_3d'] > self.selected_threshold.get()

        # [cx, cy, cz, l, w, h, rot, vx, vy]
        self.pred_bboxes = self.outputs["boxes_3d"][self.thr_idxs]
        self.pred_bboxes.tensor.detach()
        self.labels = self.outputs['labels_3d'][self.thr_idxs]

        # Extract image metas which contain, for example, the lidar to camera projection matrices
        self.img_metas = self.data["img_metas"][0]._data[0][0]

        # Extract the 6 camera images from the data and remove the padded pixels
        imgs = self.data["img"][0]._data[0].numpy()[0]
        # imgs = imgs.transpose(0, 2, 3, 1)
        imgs = imgs.transpose(0, 2, 3, 1)[:, :self.ObjectDetector.ori_shape[0], :self.ObjectDetector.ori_shape[1], :]  # [num_cams x height x width x channels]
        
        # Denormalize the images
        img_norm_cfg = self.ObjectDetector.cfg.get('img_norm_cfg')
        mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
        std = np.array(img_norm_cfg["std"], dtype=np.float32)

        for i in range(len(imgs)):
            imgs[i] = mmcv.imdenormalize(imgs[i], mean, std, to_bgr=False)
        self.imgs = imgs.astype(np.uint8)

        if initialize_bboxes and not self.video_gen_bool:
            self.update_objects_list()
            self.initialize_bboxes()


    def add_separator(self, sep="|"):
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

    def select_data_idx(self, type=0):
        # Type 0: data index; type 1: video lenght; type 2: frame rate
        popup = tk.Toplevel(self)
        popup.geometry("80x50")

        self.entry = tk.Entry(popup, width=20)
        self.entry.pack()

        button = tk.Button(popup, text="OK", command=lambda: self.close_entry(popup, type))
        button.pack()

    def close_entry(self, popup, type):
        idx = self.entry.get()
        if type == 0:
            if idx.isnumeric() and int(idx) <= (len(self.ObjectDetector.dataset)-1):
                self.data_idx = int(idx)
                self.update_info_label()
            else:
                self.show_message(f"Insert an integer between 0 and {len(self.ObjectDetector.dataset)}")
        elif type == 1:
            if idx.isnumeric() and 2 <= int(idx) <= ((len(self.ObjectDetector.dataset)-1) - self.data_idx):
                self.video_length = int(idx)
            else:
                self.show_message(f"Insert an integer between 2 and {len(self.ObjectDetector.dataset) - self.data_idx}") 
        elif type == 2:
            max_frame_rate = 100
            if idx.isnumeric() and 0 <= int(idx) <= (max_frame_rate):
                self.frame_rate = int(idx)
            else:
                self.show_message(f"Insert an integer between 0 and {max_frame_rate}") 

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

    def update_thr(self):
        self.BB_bool.set(True)
        self.show_labels.set(True)

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

    def initialize_bboxes(self):
        if hasattr(self, "bboxes"):
            if self.select_all_bboxes.get():
                self.single_bbox.set(False)
                for i in range(len(self.bboxes)):
                    self.bboxes[i].set(True)
            else:
                if len(self.bboxes) > 0:
                    self.bboxes[0].set(True)


    def get_camera_object(self):
        scores = self.update_scores()
        cam_obj = scores.index(max(scores))
        return cam_obj

    def check_layers(self):
        if self.selected_camera.get() == -1 or not self.show_attn.get():
            cam_obj = self.get_camera_object()
            self.selected_camera.set(cam_obj)
    
    def disable_attn(self):
        if not self.show_attn.get():
            self.overlay_bool.set(False)
            self.single_bbox_select(single_select=True)
            self.show_all_layers.set(True)

    def update_scores(self):
        scores = []
        scores_perc = []

        for camidx in range(len(self.ExplainableModel.attn_list[self.selected_layer.get()])):
            attn = self.ExplainableModel.attn_list[self.selected_layer.get()][camidx]
            attn = attn.clamp(min=0)
            score = round(attn.sum().item(), 2)
            scores.append(score)

        sum_scores = sum(scores)
        if sum_scores > 0 and not np.isnan(sum_scores):
            for i in range(len(scores)):
                score_perc = round(((scores[i]/sum_scores)*100))
                scores_perc.append(score_perc)
            return scores_perc
        else:
            return 0

    def capture(self):
        x0 = self.winfo_rootx()
        y0 = self.winfo_rooty()
        x1 = x0 + self.canvas.get_width_height()[0]
        y1 = y0 + self.canvas.get_width_height()[1]
        
        im = ImageGrab.grab((x0, y0, x1, y1))
        screenshots_path = "screenshots/"
        if not os.path.exists(screenshots_path):
            os.makedirs(screenshots_path)

        path = screenshots_path + f"{self.ObjectDetector.model_name}_{self.data_idx}"

        if os.path.exists(path+"_"+str(self.file_suffix)+".png"):
            self.file_suffix += 1
        else:
            self.file_suffix = 0

        path += "_" + str(self.file_suffix) + ".png"
        im.save(path)
        print(f"Screenshot saved in {path}\n")

    def change_theme(self):
        if self.tk.call("ttk::style", "theme", "use") == "azure-dark":
            self.tk.call("set_theme", "light")
            self.bg_color = "white"
        else:
            self.tk.call("set_theme", "dark")
            self.bg_color = self.info_label.cget("background")

        self.fig.set_facecolor(self.bg_color)
        self.canvas.draw()

    def save_video(self):
        if hasattr(self, "img_frames"):
            data = {'img_frames': self.img_frames, "video_idx": self.start_video_idx}

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
        self.start_video_idx = data["video_idx"]

        print(f"Video loaded from {video_pickle}.\n")