import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import numpy as np
import cv2


import tomli
import os
from tkinter import filedialog as fd
from mmcv.parallel import MMDataParallel

from App.Utils import random_data_idx, update_info_label
from Explainability.Attention import Attention
from App.Model import init_app

from mmcv.parallel import DataContainer as DC

from tkinter.messagebox import showinfo
from tkinter import scrolledtext
import random
from PIL import ImageGrab



class UI_baseclass(tk.Tk):
    '''
    Application User Interface
    '''
    def __init__(self):
        '''
        Tkinter initialization with model loading option.
        '''
        super().__init__()

        # Tkinter-related settings
        self.tk.call("source", "theme/azure.tcl")
        self.tk.call("set_theme", "dark")
        self.title('Explainable Transformer-based 3D Object Detector')
        self.geometry('1500x1500')
        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.canvas, self.video_canvas, self.fig, self.spec = None, None, None, None

        # Model and dataloader objects
        self.model, self.dataloader = None, None
        self.started_app = False
        self.video_length = 10
        
        # Main Tkinter menu in which all other cascade menus are added
        self.menubar = tk.Menu(self)

        # Cascade menus for loading model and selecting the GPU
        self.config(menu=self.menubar)
        file_opt, gpu_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.gpu_id = tk.IntVar()
        self.gpu_id.set(0)
        file_opt.add_command(label=" Load model", command=self.load_model())
        file_opt.add_command(label=" Load from config file", command=self.load_from_config())
        file_opt.add_separator()
        file_opt.add_cascade(label=" Gpu", menu=gpu_opt)
        message = "You need to reload the model to apply GPU change."
        for i in range(torch.cuda.device_count()):
            gpu_opt.add_radiobutton(label=f"GPU {i}", variable=self.gpu_id, value=i, command=self.show_message(message))

        self.menubar.add_cascade(label=" File", menu=file_opt)

        # Speeding up the testing
        #load_from_config(self)

    def start_app(self):
        '''
        It starts the UI after loading the model. Variables are initialized.
        '''
        # Booleans used for avoiding reloading same data so that the UI is speed up
        self.old_data_idx, self.old_thr, self.old_bbox_idx, self.old_expl_type, self.new_model = \
            None, None, None, None, None

        # Suffix used for saving screenshot of same model with different numbering
        self.file_suffix = 0

        # Tkinter frame for visualizing model and GPU info
        frame = tk.Frame(self)
        frame.pack(fill=tk.Y)
        self.info_text = tk.StringVar()
        self.info_label = tk.Label(frame, textvariable=self.info_text, anchor=tk.CENTER)
        self.info_label.bind("<Button-1>", lambda event: self.show_model_info())
        self.info_label.bind("<Enter>", lambda event: self.red_text())
        self.info_label.bind("<Leave>", lambda event: self.black_text())
        self.info_label.pack(side=tk.TOP)

        # Cascade menu for Data index
        dataidx_opt = tk.Menu(self.menubar)
        dataidx_opt.add_command(label=" Select data index", command=self.select_data_idx())
        dataidx_opt.add_command(label=" Select video length", command=self.select_data_idx(length=True))
        dataidx_opt.add_command(label=" Select random data", command=self.random_data_idx())

        # Cascade menus for Prediction threshold
        thr_opt = tk.Menu(self.menubar)
        self.selected_threshold = tk.DoubleVar()
        self.selected_threshold.set(0.5)
        values = np.arange(0.0, 1, 0.1).round(1)
        for i in values:
            thr_opt.add_radiobutton(label=i, variable=self.selected_threshold, command=self.update_thr())

        # Cascade menu for Camera
        camera_opt = tk.Menu(self.menubar)
        self.cameras = {'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5}
        self.cam_idx = [2, 0, 1, 5, 3, 4]  # Used for visualizing camera outputs properly
        self.selected_camera = tk.IntVar()
        for value, key in enumerate(self.cameras):
            camera_opt.add_radiobutton(label=key, variable=self.selected_camera, value=value, command=self.update_info_label())
        camera_opt.add_radiobutton(label="All", variable=self.selected_camera, value=-1, command=self.update_info_label())
        self.selected_camera.set(-1) # Default: visualize all cameras

        # Cascade menu for Attention layer
        layer_opt = tk.Menu(self.menubar)
        self.selected_layer = tk.IntVar()
        self.show_all_layers = tk.BooleanVar()
        for i in range(self.Attention.layers):
            layer_opt.add_radiobutton(label=i, variable=self.selected_layer, command=self.update_info_label())
        layer_opt.add_checkbutton(label="All", onvalue=1, offvalue=0, variable=self.show_all_layers)
        self.selected_layer.set(self.Attention.layers - 1)

        # Cascade menus for Explainable options
        expl_opt = tk.Menu(self.menubar)
        attn_rollout, grad_cam, grad_rollout = tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.expl_options = ["Attention Rollout", "Grad-CAM", "Gradient Rollout"]

        # Attention Rollout
        expl_opt.add_cascade(label=self.expl_options[0], menu=attn_rollout)
        self.head_types = ["mean", "min", "max"]
        self.selected_head_fusion = tk.StringVar()
        self.selected_head_fusion.set(self.head_types[2])
        self.raw_attn = tk.BooleanVar()
        self.raw_attn.set(True)
        dr_opt, hf_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.selected_discard_ratio = tk.DoubleVar()
        self.selected_discard_ratio.set(0.5)
        values = np.arange(0.0, 1, 0.1).round(1)
        for i in values:
            dr_opt.add_radiobutton(label=i, variable=self.selected_discard_ratio)
        for i in range(len(self.head_types)):
            hf_opt.add_radiobutton(label=self.head_types[i].capitalize(), variable=self.selected_head_fusion, value=self.head_types[i], command=lambda k=self: update_info_label(k))
        attn_rollout.add_cascade(label=" Head fusion", menu=hf_opt)
        attn_rollout.add_cascade(label=" Discard ratio", menu=dr_opt)
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
        self.handle_residual, self.apply_rule = tk.BooleanVar(),tk.BooleanVar()
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
        self.old_expl_type = self.expl_options[0]
        for i in range(len(self.expl_options)):
            expl_type_opt.add_radiobutton(label=self.expl_options[i], variable=self.selected_expl_type, value=self.expl_options[i], command=lambda: update_info_label(self))

        expl_opt.add_command(label="Evaluate explainability", command=self.evaluate_expl)

        # Cascade menus for object selection
        self.bbox_opt = tk.Menu(self.menubar)
        self.single_bbox = tk.BooleanVar()
        self.select_all_bboxes = tk.BooleanVar()
        self.select_all_bboxes.set(True)
        self.bbox_opt.add_checkbutton(label=" Single object", onvalue=1, offvalue=0, variable=self.single_bbox, command=self.single_bbox_select()) 
        self.bbox_opt.add_checkbutton(label=" Select all", onvalue=1, offvalue=0, variable=self.select_all_bboxes, command=self.initialize_bboxes())
        self.bbox_opt.add_separator()

        # Cascade menus for Additional options
        add_opt = tk.Menu(self.menubar)
        self.GT_bool, self.BB_bool, self.points_bool, self.show_scale, self.attn_contr, self.overlay_bool, self.show_labels, self.capture_bool, self.bbox_2d, self.dark_theme = \
            tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.BB_bool.set(True)
        self.show_labels.set(True)
        self.overlay_bool.set(True)
        self.bbox_2d.set(True)
        self.attn_contr.set(True)
        add_opt.add_checkbutton(label=" Show GT Bounding Boxes", onvalue=1, offvalue=0, variable=self.GT_bool)
        add_opt.add_checkbutton(label=" Show all Bounding Boxes", onvalue=1, offvalue=0, variable=self.BB_bool)
        add_opt.add_checkbutton(label=" Show attention scale", onvalue=1, offvalue=0, variable=self.show_scale)
        add_opt.add_checkbutton(label=" Show attention camera contributions", onvalue=1, offvalue=0, variable=self.attn_contr)
        add_opt.add_checkbutton(label=" Overlay attention on image", onvalue=1, offvalue=0, variable=self.overlay_bool)
        add_opt.add_checkbutton(label=" Show predicted labels", onvalue=1, offvalue=0, variable=self.show_labels)
        add_opt.add_checkbutton(label=" Capture output", onvalue=1, offvalue=0, variable=self.capture_bool)
        add_opt.add_checkbutton(label=" 2D bounding boxes", onvalue=1, offvalue=0, variable=self.bbox_2d)
        add_opt.add_checkbutton(label=" Dark theme", onvalue=1, offvalue=0, variable=self.dark_theme, command=self.change_theme())

        # Adding all cascade menus ro the main menubar menu
        self.add_separator(self)
        self.menubar.add_cascade(label="Data", menu=dataidx_opt)
        self.add_separator(self)
        self.menubar.add_cascade(label="Prediction threshold", menu=thr_opt)
        self.add_separator(self)
        self.menubar.add_cascade(label="Camera", menu=camera_opt)
        self.add_separator(self)
        self.menubar.add_cascade(label="Objects", menu=self.bbox_opt)
        self.add_separator(self)
        self.menubar.add_cascade(label="Layer", menu=layer_opt)
        self.add_separator(self)
        self.menubar.add_cascade(label="Explainability", menu=expl_opt)
        self.add_separator(self)
        self.menubar.add_cascade(label="Options", menu=add_opt)
        self.add_separator(self, "|")
        self.menubar.add_command(label="Visualize", command=self.visualize)

        # Create figure with a 3x3 grid
        self.fig = plt.figure()
        self.spec = self.fig.add_gridspec(3, 3)

        # Create canvas with the figure embedded in it, and update it after each visualization
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def load_from_config(self):
        with open("config.toml", mode="rb") as argsF:
            args = tomli.load(argsF)
            
        cfg_file = args["cfg_file"]
        weights_file = args["weights_file"]
        gpu_id = args["gpu_id"]

        self.load_model(self, cfg_file, weights_file, gpu_id)

    def load_model(self, cfg_file=None, weights_file=None, gpu_id=None):
        cfg_filetypes = (
            ('Config', '*.py'),
        )
        weights_filetypes = (
            ('Pickle', '*.pth'),
        )
        
        if cfg_file is None:
            cfg_file = fd.askopenfilename(
                title='Load model file',
                initialdir='/workspace/configs/submission/',
                filetypes=cfg_filetypes)
        
        if weights_file is None:
            weights_file = fd.askopenfilename(
                title='Load weights',
                initialdir='/workspace/work_dirs/checkpoints/',
                filetypes=weights_filetypes)  
        
        if gpu_id is not None:
            self.gpu_id.set(gpu_id)

        # Model configuration needs to load weights
        args = {}
        args["config"] = cfg_file
        args["checkpoint"] = weights_file
        model, dataloader, img_norm_cfg, cfg = init_app(args)
                
        self.model = MMDataParallel(model, device_ids=[self.gpu_id.get()])
        self.dataloader = dataloader
        self.Attention = Attention(self.model)
        self.img_norm_cfg = img_norm_cfg  # Used for image de-normalization
        self.model_name = os.path.splitext(os.path.basename(cfg_file))[0]
        self.dataloader_name = self.dataloader.dataset.metadata['version']
        self.class_names = self.dataloader.dataset.CLASSES
        self.cfg = cfg
        print("\nModel loaded.")
        
        self.new_model = True

        if not self.started_app:
            print("Starting app...\n")
            self.start_app()
            self.started_app = True
            random_data_idx(self)

        update_info_label(self)

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
        popup.title(f"Model {self.model_name}")

        text = scrolledtext.ScrolledText(popup, wrap=tk.WORD)
        for k, v in self.model.module.__dict__["_modules"].items():
            text.insert(tk.END, f"{k.upper()}\n", 'key')
            text.insert(tk.END, f"{v}\n\n")
            text.tag_config('key', background="yellow", foreground="red")

        text.pack(expand=True, fill='both')
        text.configure(state="disabled")

    def select_data_idx(self, length=False):
        popup = tk.Toplevel(self)
        popup.geometry("80x50")

        self.entry = tk.Entry(popup, width=20)
        self.entry.pack()

        button = tk.Button(popup, text="OK", command=lambda k=self: close_entry(k, popup, length))
        button.pack()

    def close_entry(self, popup, length):
        idx = self.entry.get()
        if not length:
            if idx.isnumeric() and int(idx) <= (len(self.dataloader)-1):
                self.data_idx = int(idx)
                update_info_label(self)
                popup.destroy()
            else:
                self.self.show_message(self, f"Insert an integer between 0 and {len(self.dataloader)-1}")
        else:
            if idx.isnumeric() and int(idx) <= ((len(self.dataloader)-1) - self.data_idx):
                self.video_length = int(idx)
                update_info_label(self)
                popup.destroy()
            else:
                self.show_message(self, f"Insert an integer between 0 and {(len(self.dataloader)-1) - self.data_idx}")       

    def random_data_idx(self):
        idx = random.randint(0, len(self.dataloader)-1)
        self.data_idx = idx
        update_info_label(self)

    def update_info_label(self, info=None, idx=None):
        if idx is None:
            idx = self.data_idx
        if info is None:
            info = f"Model: {self.model_name} | Dataloader: {self.dataloader_name} | Data index: {idx} | Mechanism: {self.selected_expl_type.get()}"
            if self.selected_camera.get() != -1 and not self.show_all_layers.get():
                info += f" | Camera {list(self.cameras.keys())[self.selected_camera.get()]} | Layer {self.selected_layer.get()}"
                if self.selected_expl_type.get() == "Attention Rollout":
                    info += f'| {self.selected_head_fusion.get().capitalize()} Head fusion'
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
            self.bbox_opt.add_checkbutton(label=f" {self.class_names[labels[i].item()].capitalize()} ({i})", onvalue=1, offvalue=0, variable=self.bboxes[i], command=lambda idx=i: single_bbox_select(self, idx))

    def single_bbox_select(self, idx=None):
        self.select_all_bboxes.set(False)
        if self.single_bbox.get():
            if idx is None:
                idx = 0
            for i in range(len(self.bboxes)):
                if i != idx:
                    self.bboxes[i].set(False)

    def initialize_bboxes(self):
        if self.single_bbox.get():
            self.single_bbox.set(False)
        if hasattr(self, "bboxes"):
            if self.select_all_bboxes.get():
                for i in range(len(self.bboxes)):
                    self.bboxes[i].set(True)
            else:
                if len(self.bboxes) > 0:
                    self.bboxes[0].set(True)

    def update_scores(self):
        scores = []
        self.scores_perc = []

        for camidx in range(len(self.attn_list)):
            attn = self.attn_list[camidx]
            score = round(attn.sum().item(), 2)
            scores.append(score)

        sum_scores = sum(scores)
        if sum_scores > 0:
            for i in range(len(scores)):
                score_perc = round(((scores[i]/sum_scores)*100))
                self.scores_perc.append(score_perc)

    def overlay_attention_on_image(img, attn):
        attn = cv2.applyColorMap(np.uint8(255 * attn), cv2.COLORMAP_JET)
        attn = np.float32(attn) 
        attn = cv2.resize(attn, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        img = attn + np.float32(img)
        img = img / np.max(img)
        return img

    def capture(self):
        x0 = self.winfo_rootx()
        y0 = self.winfo_rooty()
        x1 = x0 + self.canvas.get_width_height()[0]
        y1 = y0 + self.canvas.get_width_height()[1]
        
        im = ImageGrab.grab((x0, y0, x1, y1))
        path = f"screenshots/{self.model_name}_{self.data_idx}"

        if os.path.exists(path+"_"+str(self.file_suffix)+".png"):
            self.file_suffix += 1
        else:
            self.file_suffix = 0

        path += "_" + str(self.file_suffix) + ".png"
        im.save(path)

    def change_theme(self):
        if self.dark_theme.get():
            # Set light theme
            self.tk.call("set_theme", "dark")
        else:
            # Set dark theme
            self.tk.call("set_theme", "light")