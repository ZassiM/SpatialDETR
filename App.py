import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog as fd
from tkinter import scrolledtext


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

import torch
import numpy as np
import cv2
import random
import tomli
import os
from PIL import ImageGrab

from Attention import Attention
from other_scripts.save_model import init_app
from mmcv.parallel import MMDataParallel
from mmcv.parallel import DataContainer as DC


class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


def show_attn_on_img(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class App(tk.Tk):
        
    def __init__(self):
        super().__init__()
        
        style = ttk.Style(self)
        style.theme_use("clam")
        self.title('Attention Visualization')
        self.geometry('1500x1500')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        
        self.model, self.data_loader, self.gt_bboxes = None, None, None
        self.started_app = False

        frame = tk.Frame(self)
        frame.pack(fill=tk.Y)
        
        self.info_text = tk.StringVar()
        self.info_label = tk.Label(frame, textvariable=self.info_text, anchor=tk.CENTER)

        self.info_label.bind("<Button-1>", self.show_model_info)
        self.info_label.bind("<Enter>", self.red_text)
        self.info_label.bind("<Leave>", self.black_text)

        self.menubar = tk.Menu(self)
        self.config(menu=self.menubar)
        
        file_opt, gpu_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.gpu_id = tk.IntVar()
        self.gpu_id.set(0)
        file_opt.add_command(label="Load model", command=self.load_model)
        file_opt.add_command(label="Load from config file", command=self.load_from_config)
        file_opt.add_separator()
        file_opt.add_cascade(label="Gpu", menu=gpu_opt)
        message = "You need to reload model to apply GOU change"
        for i in range(torch.cuda.device_count()):
            gpu_opt.add_radiobutton(label=f"GPU {i}", variable=self.gpu_id, value=i, command=lambda:self.show_message(message))
        
        self.menubar.add_cascade(label="File", menu=file_opt)
        self.add_separator()
        
        # Speeding up the testing
        self.load_from_config()
        
    def show_message(self, message):
        showinfo(title=None, message=message)

    def red_text(self, event=None):
        self.info_label.config(fg="red")

    def black_text(self, event=None):
        self.info_label.config(fg="black")

    def show_model_info(self, event=None):
        popup = tk.Toplevel(self)
        popup.geometry("700x1000")
        popup.title(f"Model {self.model_name.capitalize()}")

        text = scrolledtext.ScrolledText(popup, wrap=tk.WORD)
        for k, v in self.model.module.__dict__["_modules"].items():
            text.insert(tk.END, f"{k.upper()}\n", 'key')
            text.insert(tk.END, f"{v}\n\n")
            text.tag_config('key', background="yellow", foreground="red")


        text.pack(expand=True, fill='both')
        text.configure(state="disabled")
        
    def start_app(self):  
        
        self.thr_idxs, self.imgs_bbox = [], []
        self.old_data_idx, self.old_bbox_idx, self.old_layer_idx, self.new_model, self.canvas, self.gt_bbox = None, None, None, None, None, None
        self.old_thr = -1
        self.head_fusion = "min"
        self.discard_ratio = 0.9      
        self.cam_idx = [2, 0, 1, 5, 3, 4]
        self.scores = []

        # Prediction threshold + Discard ratio  
        thr_opt, dr_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.selected_threshold, self.selected_discard_ratio = tk.DoubleVar(), tk.DoubleVar()
        self.selected_threshold.set(0.5)
        self.selected_discard_ratio.set(0.5)
        values = np.arange(0.0,1,0.1).round(1)
        for i in values:
            thr_opt.add_radiobutton(label=i, variable=self.selected_threshold, command=self.update_thr)
            dr_opt.add_radiobutton(label=i, variable=self.selected_discard_ratio)
            
        # Camera
        camera_opt = tk.Menu(self.menubar)
        self.cameras = {'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5, 'All': 6}
        self.selected_camera = tk.IntVar()
        self.selected_camera.set(0)
        for value, key in enumerate(self.cameras):
            camera_opt.add_radiobutton(label=key, variable=self.selected_camera, value=value)
        
        # Attention
        attn_opt, attn_rollout = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.head_types = ["mean", "min", "max"]
        self.selected_head_fusion = tk.StringVar()
        self.selected_head_fusion.set(self.head_types[2])
        self.raw_attn = tk.BooleanVar()
        self.raw_attn.set(True)
        attn_opt.add_cascade(label="Attention Rollout", menu=attn_rollout)
        for i in range(len(self.head_types)):
            attn_rollout.add_radiobutton(label=self.head_types[i].capitalize(), variable=self.selected_head_fusion, value=self.head_types[i])
        attn_rollout.add_radiobutton(label="All", variable=self.selected_head_fusion, value="all")
        attn_rollout.add_checkbutton(label="Raw attention", variable=self.raw_attn, onvalue=1, offvalue=0)
        attn_opt.add_radiobutton(label="Grad-CAM", variable=self.selected_head_fusion, value="gradcam")
        attn_opt.add_separator()
        attn_opt.add
                
        attn_layer = tk.Menu(self.menubar)
        self.selected_layer = tk.IntVar()
        self.selected_layer.set(5)
        for i in range(len(self.model.module.pts_bbox_head.transformer.decoder.layers)):
            attn_layer.add_radiobutton(label=i, variable=self.selected_layer)
        attn_layer.add_radiobutton(label="All", variable=self.selected_layer, value=6)
        attn_opt.add_cascade(label="Layer", menu=attn_layer)

        # Bounding boxes
        self.bbox_opt = tk.Menu(self.menubar)
        self.single_bbox = tk.BooleanVar()
        self.single_bbox.set(False)
        self.bboxes = []
        # self.bbox_idx = [0]
        self.bbox_idx = []
        self.bbox_opt.add_checkbutton(label="Single bounding box", onvalue=1, offvalue=0, variable = self.single_bbox)
        self.bbox_opt.add_separator()

        # Data index
        dataidx_opt = tk.Menu(self.menubar)
        dataidx_opt.add_command(label="Select data index", command=self.select_data_idx)
        dataidx_opt.add_command(label="Select random data", command=self.random_data_idx)

        
        # View options
        add_opt = tk.Menu(self.menubar)
        self.GT_bool, self.BB_bool, self.points_bool, self.scale, self.attn_contr, self.attn_norm, self.overlay, self.show_labels, self.showed_info, self.capture_bool= tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.BB_bool.set(True)
        self.scale.set(True)
        self.show_labels.set(True)
        self.attn_norm.set(True)
        self.attn_contr.set(True)
        self.showed_info.set(False)
        self.capture_bool.set(True)
        add_opt.add_checkbutton(label="Show GT Bounding Boxes", onvalue=1, offvalue=0, variable=self.GT_bool)
        add_opt.add_checkbutton(label="Show all Bounding Boxes", onvalue=1, offvalue=0, variable=self.BB_bool)
        add_opt.add_checkbutton(label="Show attention scale", onvalue=1, offvalue=0, variable=self.scale)
        add_opt.add_checkbutton(label="Show attention camera contributions", onvalue=1, offvalue=0, variable=self.attn_contr)
        add_opt.add_checkbutton(label="Normalize attention", onvalue=1, offvalue=0, variable=self.attn_norm)
        add_opt.add_checkbutton(label="Overlay attention on image", onvalue=1, offvalue=0, variable=self.overlay)
        add_opt.add_checkbutton(label="Show predicted labels", onvalue=1, offvalue=0, variable=self.show_labels)
        add_opt.add_checkbutton(label="Capture output", onvalue=1, offvalue=0, variable=self.capture_bool)
        add_opt.add_checkbutton(label="Show info", command=self.show_info)


        self.menubar.add_cascade(label="Data", menu=dataidx_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Prediction threshold", menu=thr_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Discard ratio", menu=dr_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Camera", menu=camera_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Attention", menu=attn_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Bounding boxes", menu=self.bbox_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Options", menu=add_opt)
        self.add_separator("|")
        self.menubar.add_command(label="Visualize", command = self.visualize)

            
    def add_separator(self, sep="|"):
        self.menubar.add_command(label=sep, activebackground=self.menubar.cget("background"))
        #\u007C, \u22EE´

    def select_data_idx(self):
        popup = tk.Toplevel(self)
        popup.geometry("50x50")

        self.entry = tk.Entry(popup, width=20)
        self.entry.pack()

        button = tk.Button(popup, text="OK", command=lambda:self.close_entry(popup))
        button.pack()

    def close_entry(self, popup):
        idx = self.entry.get()
        if idx.isnumeric() and int(idx) <= (len(self.data_loader)-1):
            self.data_idx = int(idx)
            self.update_data_label(int(idx))
            popup.destroy()
        else:
            self.show_message(f"Insert an integer between 0 and {len(self.data_loader)-1}")

    def capture(self):
        x0 = self.winfo_rootx()
        y0 = self.winfo_rooty()
        x1 = x0 + self.canvas.get_width_height()[0]
        y1 = y0 + self.canvas.get_width_height()[1]
        
        im = ImageGrab.grab((x0, y0, x1, y1))
        self.suffix = 0
        path = f"screenshots/{self.model_name}_{self.data_idx}"

        if os.path.exists(path+".png"):
            self.suffix += 1
        else:
            self.suffix = 0

        path += "_" + str(self.suffix) + ".png"
        im.save(path) # Can also say im.show() to display it

    def update_data_label(self, idx):
        info = f"Model name: {self.model_name} | GPU ID: {self.gpu_id.get()} | Data index: {int(idx)}"
        self.info_text.set(info)
        
    def load_from_config(self):

        with open("config.toml", mode="rb") as argsF:
            args = tomli.load(argsF)
            
        cfg_file = args["cfg_file"]
        weights_file = args["weights_file"]
        gpu_id = args["gpu_id"]

        self.load_model(cfg_file, weights_file, gpu_id)
        
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
        model, dataloader, checkpoint = init_app(args)
                
        self.model = MMDataParallel(model, device_ids=[self.gpu_id.get()])
        self.data_loader = dataloader
        self.gen = Attention(self.model)
        self.new_model = True
        self.model_name = os.path.splitext(os.path.basename(cfg_file))[0]

        self.random_data_idx()

        if not self.started_app:
            self.start_app()
            self.started_app = True
        
        print("Loading completed.")

    def show_info(self):
        if not self.showed_info.get():
            self.info_label.pack(side=tk.TOP)
            self.showed_info.set(True)
        else:
            self.info_label.pack_forget()
            self.showed_info.set(False)

    def random_data_idx(self):
        idx = random.randint(0, len(self.data_loader)-1)
        self.data_idx = idx
        self.update_data_label(idx)
        
    def update_thr(self):
        self.BB_bool.set(True)
        self.show_labels.set(True)
        


    def update_scores(self):
        self.all_attn = self.gen.get_all_attn(self.bbox_idx, self.nms_idxs, self.head_fusion, self.discard_ratio, self.raw_attn.get())
        self.scores = []
        self.scores_perc = []
        if self.selected_layer.get() == 6:
            for layer in range(6):
                attn = self.all_attn[layer][self.selected_camera.get()]
                score = round(attn.sum().item(), 2)
                self.scores.append(score)
        else:
            for cam in range(6):
                attn = self.all_attn[self.selected_layer.get()][cam]
                score = round(attn.sum().item(), 2)
                self.scores.append(score)
        
        sum_scores = sum(self.scores)
        for i in range(len(self.scores)):
            score_perc = round(((self.scores[i]/sum_scores)*100))
            self.scores_perc.append(score_perc)
                
    def show_attn_maps(self, grid_clm=1):
        
        if self.attn_contr.get():
            self.update_scores()
            
        fontsize = 8
        attn_cameras = []
        if self.selected_camera.get() == 6:
            for i in range(6):
                attn_cam = self.gen.generate_rollout(self.bbox_idx, self.nms_idxs, i, self.head_fusion, self.discard_ratio, self.raw_attn.get())       
                attn_cameras.append(attn_cam)
            attn_max = torch.max(torch.cat(attn_cameras))
            self.gen.camidx = self.selected_camera.get()
            
        if self.selected_layer.get() == 6 or self.selected_camera.get() == 6:
            layer_grid = self.spec[1,grid_clm].subgridspec(2,3)
            for i in range(6):
                if self.selected_layer.get() == 6: self.gen.layer = i
                else: self.selected_camera.set(self.cam_idx[i])
                attn = self.gen.generate_rollout(self.bbox_idx, self.nms_idxs, self.selected_camera.get(), self.head_fusion, self.discard_ratio, self.raw_attn.get())
                # Normalization
                if attn_cameras and self.attn_norm.get():
                    attn /= attn_max
                attn = attn.view(29, 50).cpu().numpy()
                ax_attn = self.fig.add_subplot(layer_grid[i>2,i if i<3 else i-3])
                if attn_cameras and self.attn_norm.get():
                    attmap = ax_attn.imshow(attn, vmin=0, vmax=1)
                else:
                    attmap = ax_attn.imshow(attn)
                ax_attn.axis('off')
                if self.scale.get():  
                    im_ratio = attn.shape[1]/attn.shape[0]
                    self.fig.colorbar(attmap, ax=ax_attn, orientation='horizontal', extend='both', fraction=0.047*im_ratio)
                if self.attn_contr.get():
                    if self.selected_layer.get() == 6: ax_attn.set_title(f'{list(self.cameras.keys())[self.selected_camera.get()]}, layer {self.gen.layer}, {self.head_fusion}, {self.scores_perc[i]}%', fontsize=fontsize)
                    else: ax_attn.set_title(f'{list(self.cameras.keys())[self.selected_camera.get()]}, layer {self.gen.layer}, {self.head_fusion}, {self.scores_perc[self.cam_idx[i]]}%', fontsize=fontsize)
                else:
                    ax_attn.set_title(f'{list(self.cameras.keys())[self.selected_camera.get()]}, layer {self.gen.layer}, {self.head_fusion}', fontsize=fontsize)

            if self.selected_layer.get() != 6:
                self.selected_camera.set(6)
                
        else:
            attn = self.gen.generate_rollout(self.bbox_idx, self.nms_idxs, self.selected_camera.get(), self.head_fusion, self.discard_ratio, self.raw_attn.get())
            attn = attn.view(29, 50).cpu().numpy()
            ax_attn = self.fig.add_subplot(self.spec[1,grid_clm])
            if self.attn_norm.get():
                if attn.max() != 0: attn /= attn.max()
                attmap = ax_attn.imshow(attn, vmin=0, vmax=1)
            else:
                attmap = ax_attn.imshow(attn)
            ax_attn.axis('off')
            ax_attn.set_title(f'{list(self.cameras.keys())[self.selected_camera.get()]}, layer {self.gen.layer}, {self.head_fusion}, {self.scores_perc[self.selected_camera.get()]}%')
            if self.scale.get():  
                im_ratio = attn.shape[1]/attn.shape[0]
                self.fig.colorbar(attmap, ax=ax_attn, orientation='horizontal', extend='both', fraction=0.047*im_ratio)
        
    def update_bbox(self):
        self.bboxes = []
        self.bbox_opt.delete(3, 'end')
        for i in range(len(self.thr_idxs.nonzero())):
            view_bbox = tk.BooleanVar()
            view_bbox.set(False)
            self.bboxes.append(view_bbox)
            self.bbox_opt.add_checkbutton(label=f"{class_names[self.labels[i].item()].capitalize()} ({i})", onvalue=1, offvalue=0, variable=self.bboxes[i], command=lambda idx=i: self.single_bbox_select(idx))

    def single_bbox_select(self, idx):
        if self.single_bbox.get():
            for i in range(len(self.bboxes)):
                if i!= idx: 
                    self.bboxes[i].set(False)


    def update_values(self):
        if isinstance(self.data_loader, list):
            self.data = self.data_loader[self.data_idx]
        else:
            data = self.data_loader.dataset[self.data_idx]
            metas = [[data['img_metas'][0].data]]
            img = [data['img'][0].data.unsqueeze(0)]
            data['img_metas'][0] = DC(metas, cpu_only = True)
            data['img'][0] = DC(img)
            self.data = data

        if self.selected_head_fusion.get() != "gradcam":
            outputs = self.gen.extract_attentions(self.data)
        else:
            outputs = self.gen.extract_attentions(self.data, self.bbox_idx)

        self.nms_idxs = self.model.module.pts_bbox_head.bbox_coder.get_indexes() 
        self.outputs = outputs[0]["pts_bbox"]
        
        self.thr_idxs = self.outputs['scores_3d'] > self.selected_threshold.get()
        self.pred_bboxes = self.outputs["boxes_3d"][self.thr_idxs]
        self.pred_bboxes.tensor.detach()
        self.labels = self.outputs['labels_3d'][self.thr_idxs]
        
        imgs = self.data["img"][0]._data[0].numpy()[0]
        imgs = imgs.transpose(0,2,3,1)[:,:900,:,:]
        self.imgs = imgs.astype(np.uint8)
 
        self.img_metas = self.data["img_metas"][0]._data[0][0]
    
    def visualize(self):
        
        self.fig = plt.figure(figsize=(80,60), layout="constrained")
        self.spec = self.fig.add_gridspec(3, 3)
        
        # Avoiding to visualize all layers and all cameras at the same time
        if self.selected_camera.get() == 6 and self.selected_layer.get() == 6: self.selected_layer.set(5)
        
        if self.old_data_idx != self.data_idx or self.old_thr != self.selected_threshold.get() or self.new_model:
            self.update_values()
            self.update_bbox()
            self.bboxes[0].set(True)  # Default bbox for first visualization
            if self.new_model: self.new_model = False 
            if self.old_thr != self.selected_threshold.get(): self.old_thr = self.selected_threshold.get()
            if self.old_data_idx != self.data_idx: self.old_data_idx = self.data_idx
        
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        if self.selected_layer.get() != 6:
            self.gen.layer = self.selected_layer.get()
            

        if self.GT_bool.get():
            self.gt_bbox = self.data_loader.dataset.get_ann_info(self.data_idx)['gt_bboxes_3d']
            
        else:
            self.gt_bbox = None

        self.head_fusion = self.selected_head_fusion.get()
        self.discard_ratio = self.selected_discard_ratio.get()
        
        self.imgs_bbox = []
        for camidx in range(6):
            img = draw_lidar_bbox3d_on_img(
                    self.pred_bboxes ,
                    self.imgs[camidx],
                    self.img_metas['lidar2img'][camidx],
                    self.img_metas,
                    color=(0,255,0),
                    with_label = self.show_labels.get(),
                    all_bbx = self.BB_bool.get(),
                    bbx_idx = self.bbox_idx)  
            
            if self.gt_bbox:
                img = draw_lidar_bbox3d_on_img(
                        self.gt_bbox,
                        img,
                        self.img_metas['lidar2img'][camidx],
                        self.img_metas,
                        color=(255,0,0))
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs_bbox.append(img)
                       
            
        if self.head_fusion not in ("all", "gradcam"):
            self.show_attn_maps()

        elif self.head_fusion == "gradcam":   
            self.gen.extract_attentions(self.data, self.bbox_idx)
            attn = self.gen.generate_attn_gradcam(self.bbox_idx, self.nms_idxs, self.selected_camera.get())
            attn = attn.view(29, 50).cpu().numpy()
            ax_attn = self.fig.add_subplot(self.spec[1,1])
            attmap = ax_attn.imshow(attn)
            ax_attn.axis('off')
            ax_attn.set_title(f'{list(self.cameras.keys())[self.selected_camera.get()]}')  
            
            if self.scale.get():  
                im_ratio = attn.shape[1]/attn.shape[0]
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                self.fig.colorbar(attmap, norm=norm, ax=ax_attn, orientation='horizontal', fraction=0.047*im_ratio)
        
        elif self.head_fusion == "all":
            for k in range(len(self.head_types)):
                self.head_fusion = self.head_types[k]
                self.show_attn_maps(grid_clm = k)
                
       
        for i in range(6):
            if i < 3:
                ax = self.fig.add_subplot(self.spec[0, i])
            else:
                ax = self.fig.add_subplot(self.spec[2,i-3])

            ax.imshow(self.imgs_bbox[self.cam_idx[i]])
            ax.axis('off')
            
            if self.attn_contr.get():
                if self.selected_layer.get() == 6: ax.set_title(f'{list(self.cameras.keys())[self.cam_idx[i]]}') 
                else: ax.set_title(f'{list(self.cameras.keys())[self.cam_idx[i]]}, {self.scores_perc[self.cam_idx[i]]}%') 
            else: 
                ax.set_title(f'{list(self.cameras.keys())[self.cam_idx[i]]}')
   
            
        if self.canvas: self.canvas.get_tk_widget().pack_forget()
        self.canvas = FigureCanvasTkAgg(self.fig, self)  
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        if self.capture_bool.get():
            self.capture()
        

        