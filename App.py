from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog as fd

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

import torch
import numpy as np
import cv2
import random
import pathlib
import tomli


from Attention import Generator
from other_scripts.save_model import init_app
from mmcv.parallel import MMDataParallel
from matplotlib import gridspec
import matplotlib.pyplot as plt

            
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



class App(Tk):
        
    def __init__(self):
        super().__init__()
        
        style = ttk.Style(self)
        style.theme_use("clam")
        self.title('Attention Visualization')
        self.geometry('1500x1500')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        
        self.model, self.dataset, self.gt_bboxes = None, None, None
        self.started_app = False
        self.menubar = Menu(self)
        self.config(menu=self.menubar)
        
        file_opt, gpu_opt = Menu(self.menubar), Menu(self.menubar)
        self.gpu_id = IntVar()
        self.gpu_id.set(0)
        file_opt.add_command(label="Load model", command = self.load_model)
        #file_opt.add_command(label="Load weights", command = self.load_weights)
        file_opt.add_command(label="Load dataset", command = self.load_dataset)
        file_opt.add_command(label="Load gt bboxes", command = self.load_gtbboxes)
        file_opt.add_command(label="Load from config file", command = self.load_from_config)
        file_opt.add_separator()
        file_opt.add_cascade(label="Gpu", menu=gpu_opt)
        for i in range(torch.cuda.device_count()):
            gpu_opt.add_radiobutton(label = f"GPU {i}", variable = self.gpu_id, value = i, command = self.show_info)
        
        self.menubar.add_cascade(label="File", menu=file_opt)
        self.add_separator()
        
        # Speeding up the testing
        #self.load_from_config()
        
    def show_info(self):
        showinfo(title=None, message="Reload model to apply the GPU change.")
        
    def start_app(self):  
        
        self.thr_idxs, self.imgs_bbox  = [], []
        self.old_data_idx, self.old_bbox_idx, self.old_layer_idx, self.new_model, self.canvas, self.gt_bbox= None, None, None, None, None, None
        self.old_thr = -1
        self.head_fusion = "min"
        self.discard_ratio = 0.9      
        self.cam_idx = [2, 0, 1, 5, 3, 4]
        self.scores = []
        
        frame = Frame(self)
        frame.pack(fill=Y)
        
        self.data_label = StringVar()
        self.data_label.set("Select data index:")
        label0 = Label(frame,textvariable=self.data_label, anchor = CENTER)
        label0.pack(side=TOP)
        self.data_idx = Scale(frame, from_=0, to=len(self.dataset)-1, showvalue=0, orient=HORIZONTAL, command = self.update_data_label)
        idx = random.randint(0, len(self.dataset)-1)
        self.data_idx.set(idx)
        self.data_idx.pack()
        
        frame1 = Frame(self)
        frame1.pack(fill=Y)
        
        self.text_label = StringVar()
        self.text_label.set("Select bbox index:")
        label2 = Label(frame1,textvariable = self.text_label)
        label2.pack(side=TOP)
        self.selected_bbox = Scale(frame1, from_=0, to=len(self.thr_idxs), orient=HORIZONTAL, showvalue=0, command = self.update_bbox_label)
        self.selected_bbox.set(0)
        self.selected_bbox.pack()
        
        
        # Prediction threshold + Discard ratio  
        thr_opt, dr_opt = Menu(self.menubar), Menu(self.menubar)
        self.selected_threshold, self.selected_discard_ratio = DoubleVar(), DoubleVar()
        self.selected_threshold.set(0.5)
        self.selected_discard_ratio.set(0.5)
        values = np.arange(0.0,1,0.1).round(1)
        for i in values:
            thr_opt.add_radiobutton(label=i, variable=self.selected_threshold, command = self.update_thr)
            dr_opt.add_radiobutton(label=i, variable=self.selected_discard_ratio)
            
        # Camera
        camera_opt = Menu(self.menubar)
        self.cameras = {'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5, 'All': 6}
        self.selected_camera = IntVar()
        self.selected_camera.set(0)
        for value,key in enumerate(self.cameras):
            camera_opt.add_radiobutton(label = key, variable = self.selected_camera, value = value)
        
        # Attention
        attn_opt, attn_rollout = Menu(self.menubar), Menu(self.menubar)
        self.head_types = ["mean", "min", "max"]
        self.selected_head_fusion = StringVar()
        self.selected_head_fusion.set(self.head_types[2])
        self.raw_attn = BooleanVar()
        self.raw_attn.set(True)
        attn_opt.add_cascade(label="Attention Rollout", menu=attn_rollout)
        for i in range(len(self.head_types)):
            attn_rollout.add_radiobutton(label = self.head_types[i].capitalize(), variable = self.selected_head_fusion, value = self.head_types[i])
        attn_rollout.add_radiobutton(label = "All", variable = self.selected_head_fusion, value = "all")
        attn_rollout.add_checkbutton(label = "Raw attention", variable = self.raw_attn, onvalue=1, offvalue=0)
        attn_opt.add_radiobutton(label = "Grad-CAM", variable = self.selected_head_fusion, value = "gradcam")
        attn_opt.add_separator()
                
        attn_layer = Menu(self.menubar)
        self.selected_layer = IntVar()
        self.selected_layer.set(5)
        for i in range(len(self.model.module.pts_bbox_head.transformer.decoder.layers)):
            attn_layer.add_radiobutton(label = i, variable = self.selected_layer)
        attn_layer.add_radiobutton(label = "All", variable = self.selected_layer, value=6)
        attn_opt.add_cascade(label="Layer", menu=attn_layer)
        
        # View options
        add_opt = Menu(self.menubar)
        self.GT_bool, self.BB_bool, self.points_bool, self.scale, self.attn_contr, self.attn_norm, self.overlay, self.show_labels = BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar()
        self.BB_bool.set(True)
        self.scale.set(True)
        self.show_labels.set(True)
        self.attn_contr.set(True)
        add_opt.add_checkbutton(label="Show GT Bounding Boxes", onvalue=1, offvalue=0, variable=self.GT_bool)
        add_opt.add_checkbutton(label="Show all Bounding Boxes", onvalue=1, offvalue=0, variable=self.BB_bool)
        add_opt.add_checkbutton(label="Show attention scale", onvalue=1, offvalue=0, variable=self.scale)
        add_opt.add_checkbutton(label="Show attention camera contributions", onvalue=1, offvalue=0, variable=self.attn_contr)
        add_opt.add_checkbutton(label="Normalize attention", onvalue=1, offvalue=0, variable=self.attn_norm)
        add_opt.add_checkbutton(label="Overlay attention on image", onvalue=1, offvalue=0, variable=self.overlay)
        add_opt.add_checkbutton(label="Show predicted labels", onvalue=1, offvalue=0, variable=self.show_labels)
    
        self.menubar.add_cascade(label="Prediction threshold", menu=thr_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Discard ratio", menu=dr_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Camera", menu=camera_opt)
        self.add_separator()
        self.menubar.add_cascade(label="Attention", menu=attn_opt)
        self.add_separator()
        self.menubar.add_cascade(label="View", menu=add_opt)

        plot_button = Button(self, command = self.visualize, text = "Visualize")
        
        plot_button.pack()
        
        
    def load_from_config(self):

        with open("config.toml", mode = "rb") as argsF:
            args = tomli.load(argsF)
            
        model_filename = args["model_filename"]
        print(f"\nLoading Model from {model_filename}...\n")
        model = torch.load(open(model_filename, 'rb'))

        dataset_filename = args["dataset_filename"]
        print(f"Loading Dataset from {dataset_filename}...\n")
        dataset = torch.load(open(dataset_filename, 'rb'))

        
        GT_filename = args["GTbboxes_filename"]
        print(f"Loading GT Bounding Boxes from {GT_filename}...\n")
        gt_bboxes = torch.load(open(GT_filename, 'rb'))
        
        self.model = MMDataParallel(model, device_ids = [self.gpu_id.get()])
        self.dataset = dataset
        self.gt_bboxes = gt_bboxes
        self.gen = Generator(self.model)
        
        if not self.started_app:
            self.start_app()
            self.started_app = True
        
        print("Loading completed.")

        
    def load_model(self):
        filetypes = (
            ('Config', '*.py'),
            ('Pickle', '*.pth'),
        )
        
        filename = fd.askopenfilename(
            title='Load model file',
            initialdir='/workspace/configs/submission/frozen_4/',
            filetypes=filetypes)
        
        if pathlib.Path(filename).suffix == '.pth':
            model = torch.load(open(filename, 'rb'))
            
        elif pathlib.Path(filename).suffix == '.py':
            # Model configuration needs to load weights
            args={}
            args["config"] = filename
            args["checkpoint"] = self.load_weights()
            model, _, dataloader = init_app(args)
                
        self.model = MMDataParallel(model, device_ids = [self.gpu_id.get()])
        #self.dataset = list(dataloader)
        self.gen = Generator(self.model)
        self.new_model = True
        
        if not self.started_app:
            self.load_dataset()
            self.load_gtbboxes()
            self.start_app()
            self.started_app = True
        
        print("Loading completed.")
            
    def load_weights(self):
        filetypes = (
            ('Pickle', '*.pth'),
        )

        filename = fd.askopenfilename(
            title='Load weights',
            initialdir='/workspace/work_dirs/checkpoints/',
            filetypes=filetypes)     
                
        return filename

    
    def load_dataset(self):
        filetypes = (
            ('Pickle', '*.pth'),
        )
        filename = fd.askopenfilename(
            title='Load dataset',
            initialdir='/workspace/work_dirs/saved/',
            filetypes=filetypes)
        
        self.dataset = torch.load(open(filename, 'rb'))
        print("Loading completed.")
        
    def load_gtbboxes(self):
        filetypes = (
            ('Pickle', '*.pth'),
        )

        filename = fd.askopenfilename(
            title='Load GT bboxes',
            initialdir='/workspace/work_dirs/saved/',
            filetypes=filetypes)

        self.gt_bboxes = torch.load(open(filename, 'rb'))
        print("Loading completed.")
        
    def add_separator(self):
        self.menubar.add_command(label="\u22EE", activebackground=self.menubar.cget("background"))

    def update_thr(self):
        self.BB_bool.set(True)
        self.show_labels.set(True)
        
    def update_data_label(self, idx):
       self.data_label.set(f"Select data index: {int(idx)}")

    def update_bbox_label(self, idx):
       self.text_label.set(f"Select bbox index: {class_names[self.labels[int(idx)].item()]} ({int(idx)})")

    def update_values(self):
        self.data = self.dataset[self.data_idx.get()]
        if self.selected_head_fusion.get() != "gradcam":
            outputs = self.gen.extract_attentions(self.data)
        else:
            outputs = self.gen.extract_attentions(self.data, self.selected_bbox.get())

        self.nms_idxs = self.model.module.pts_bbox_head.bbox_coder.get_indexes() 
        self.outputs = outputs[0]["pts_bbox"]
        
        imgs = self.data["img"][0]._data[0].numpy()[0]
        imgs = imgs.transpose(0,2,3,1)[:,:900,:,:]
        self.imgs = imgs.astype(np.uint8)
 
        self.img_metas = self.data["img_metas"][0]._data[0][0]
        
    def update_scores(self):
        self.all_attn = self.gen.get_all_attn(self.selected_bbox.get(), self.nms_idxs, self.head_fusion, self.discard_ratio, self.raw_attn.get())
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
                
    def show_attn_maps(self, grid_clm = 1):
        
        if self.attn_contr.get():
            self.update_scores()
            
        fontsize = 8
        attn_cameras = []
        if self.selected_camera.get() == 6:
            for i in range(6):
                attn_cam = self.gen.generate_rollout(self.selected_bbox.get(), self.nms_idxs, i, self.head_fusion, self.discard_ratio, self.raw_attn.get())       
                attn_cameras.append(attn_cam)
            attn_max = torch.max(torch.cat(attn_cameras))
            self.gen.camidx = self.selected_camera.get()
            
        if self.selected_layer.get() == 6 or self.selected_camera.get() == 6:
            layer_grid = self.spec[1,grid_clm].subgridspec(2,3)
            for i in range(6):
                if self.selected_layer.get() == 6: self.gen.layer = i
                else: self.selected_camera.set(self.cam_idx[i])
                attn = self.gen.generate_rollout(self.selected_bbox.get(), self.nms_idxs, self.selected_camera.get(), self.head_fusion, self.discard_ratio, self.raw_attn.get())
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
            attn = self.gen.generate_rollout(self.selected_bbox.get(), self.nms_idxs, self.selected_camera.get(), self.head_fusion, self.discard_ratio, self.raw_attn.get())
            attn = attn.view(29, 50).cpu().numpy()
            ax_attn = self.fig.add_subplot(self.spec[1,grid_clm])
            if self.attn_norm.get():
                attn /= attn.max()
                attmap = ax_attn.imshow(attn, vmin=0, vmax=1)
            else:
                attmap = ax_attn.imshow(attn)
            ax_attn.axis('off')
            ax_attn.set_title(f'{list(self.cameras.keys())[self.selected_camera.get()]}, layer {self.gen.layer}, {self.head_fusion}, {self.scores_perc[self.selected_camera.get()]}%')
            if self.scale.get():  
                im_ratio = attn.shape[1]/attn.shape[0]
                self.fig.colorbar(attmap, ax=ax_attn, orientation='horizontal', extend='both', fraction=0.047*im_ratio)
        
    
    
    def visualize(self):
        
        self.fig = plt.figure(figsize=(80,60), layout="constrained")
        self.spec = self.fig.add_gridspec(3, 3)
        
        # Avoiding to visualize all layers and all cameras at the same time
        if self.selected_camera.get() == 6 and self.selected_layer.get() == 6:
            if self.selected_camera.get() == 6: self.selected_layer.set(5)
            if self.selected_layer.get() == 6: self.selected_camera.set(0)
        
        if self.old_data_idx != self.data_idx.get() or self.old_thr != self.selected_threshold.get() or self.new_model:
            self.old_data_idx = self.data_idx.get()
            self.update_values()
            self.thr_idxs = self.outputs['scores_3d'] > self.selected_threshold.get()
            self.selected_bbox.configure(to = len(self.thr_idxs.nonzero())-1)
            self.selected_bbox.set(0)
            if self.new_model: self.new_model = False 
            if self.old_thr != self.selected_threshold.get(): self.old_thr = self.selected_threshold.get()

        self.labels = self.outputs['labels_3d'][self.thr_idxs]
        self.pred_bboxes = self.outputs["boxes_3d"][self.thr_idxs]
        self.pred_bboxes.tensor.detach()
            
        if self.selected_layer.get() != 6:
            self.gen.layer = self.selected_layer.get()
            

        if self.GT_bool.get():
            self.gt_bbox = self.gt_bboxes[self.data_idx.get()]
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
                    bbx_idx = self.selected_bbox.get())  
            
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
            # if self.overlay.get():
            #     attn = cv2.resize(attn, (self.imgs_bbox[0].shape[1], self.imgs_bbox[0].shape[0]))
            #     img = show_attn_on_img(self.imgs_bbox[self.selected_camera.get()], attn)
            #     attmap = ax_attn.imshow(img)
            
            #else
            self.show_attn_maps()


        elif self.head_fusion == "gradcam":   
            self.gen.extract_attentions(self.data, self.selected_bbox.get())
            attn = self.gen.generate_attn_gradcam(self.selected_bbox.get(), self.nms_idxs, self.selected_camera.get())
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
        

        