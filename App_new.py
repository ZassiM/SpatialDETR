from tkinter import *
from tkinter import ttk
from tkinter import font

import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl

#matplotlib.use('Agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

import torch
import numpy as np
import cv2
import random


from Attention import Generator

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
        
    def __init__(self, model, data_loader, gt_bboxes):
        super().__init__()
        
        
        style = ttk.Style(self)
        style.theme_use("clam")
        
        app_font = font.nametofont("TkDefaultFont")  # Get default font value into Font object
        
        app_font.actual()
        
        self.title('Attention Visualization')
        self.geometry('1500x1500')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1

        self.model = model
        self.data_loader = data_loader
        self.gt_bboxes = gt_bboxes
        self.gt_bbox = None
        
        self.gen = Generator(self.model)
        
        self.canvas = None
        self.thr_idxs = []
        self.imgs_bbox = []
        self.old_data_idx = None
        self.old_bbox_idx = None
        self.old_layer_idx = None
        self.old_thr = -1

        # label0 = Label(text="Select data index:", anchor = CENTER)
        # label0.pack(fill=X, padx=5, pady=5)
        # self.data_idx = Scale(self, from_=0, to=len(self.data_loader)-1, orient=HORIZONTAL)
        # idx = random.randint(0, len(self.data_loader)-1)
        # self.data_idx.set(idx)
        # self.data_idx.pack()
        
        self.head_fusion = "min"
        self.discard_ratio = 0.9        


        self.thr_text = StringVar()
        self.thr_text.set("Select prediction threshold:")
        label5 = Label(textvariable = self.thr_text , anchor = CENTER)
        label5.pack(fill=X, padx=5, pady=5)
        self.selected_threshold = Scale(self, from_=0, to=1, showvalue = 0, resolution = 0.1, orient=HORIZONTAL, command = self.update_thr)
        self.selected_threshold.set(0.5)
        self.selected_threshold.pack()
            
        self.text_label = StringVar()
        self.text_label.set("Select bbox index:")
        label2 = Label(textvariable = self.text_label)
        label2.pack(fill=X, padx=5, pady=5)
        self.selected_bbox = Scale(self, from_=0, to=len(self.thr_idxs), showvalue = 0, orient=HORIZONTAL, command = self.update_class)
        self.selected_bbox.set(0)
        self.selected_bbox.pack()
        
        # self.layer_text = StringVar()
        # self.layer_text.set("Select layer to visualize:")
        # label10 = Label(textvariable=self.layer_text, anchor = CENTER)
        # label10.pack(fill=X, padx=5, pady=5)
        # self.attn_layer = Scale(self, from_=0, to=5, orient=HORIZONTAL, showvalue = False, command = self.update_layer)
        # self.attn_layer.set(5)
        # self.attn_layer.pack()
        
        # label3 = Label(text = "Select head fusion mode:")
        # label3.pack(fill=X, padx=5, pady=5)
        # frame1 = Frame(self)
        # frame1.pack()
        # self.head_types = ["mean", "min", "max"]
        # self.selected_head_fusion = StringVar()
        # self.selected_head_fusion.set(self.head_types[0])
        # for i in range(len(self.head_types)):
        #     Radiobutton(frame1, text = self.head_types[i].capitalize(), variable = self.selected_head_fusion, value = self.head_types[i]).grid(row=0,column=i)
        # Radiobutton(frame1, text = "All", variable = self.selected_head_fusion, value = "all").grid(row=0,column=i+1)
        # Radiobutton(frame1, text = "Grad-CAM", variable = self.selected_head_fusion, value = "gradcam").grid(row=0,column=i+2)

        # self.raw_attn = IntVar()
        # self.raw_attn.set(1)
        # Checkbutton(frame1, text='Raw attention',variable=self.raw_attn, onvalue=1, offvalue=0).grid(row=0,column=i+3)

        
        # self.dr_text = StringVar()
        # self.dr_text.set("Select discard ratio:")
        # label4 = Label(textvariable = self.dr_text, anchor = CENTER)
        # label4.pack(fill=X, padx=5, pady=5)
        # self.selected_discard_ratio = Scale(self, from_=0, to=0.9, showvalue = 0, resolution = 0.1, orient=HORIZONTAL, command = self.update_dr)
        # self.selected_discard_ratio.set(0)
        # self.selected_discard_ratio.pack()
        
        menubar = Menu(self)
        self.config(menu=menubar)

        camera_opt = Menu(menubar)
        
        self.cameras = {'FRONT': 0, 'FRONT-RIGHT': 1, 'FRONT-LEFT': 2, 'BACK': 3, 'BACK-LEFT': 4, 'BACK-RIGHT': 5}
        self.selected_camera = IntVar()
        for value,key in enumerate(self.cameras):
            camera_opt.add_radiobutton(label = key, variable = self.selected_camera, value = value)

        attn_opt = Menu(menubar)

        attn_rollout = Menu(menubar)
        self.head_types = ["mean", "min", "max"]
        self.selected_head_fusion = StringVar()
        self.selected_head_fusion.set(self.head_types[0])
        attn_opt.add_cascade(label="Attention Rollout", menu=attn_rollout)
        for i in range(len(self.head_types)):
            attn_rollout.add_radiobutton(label = self.head_types[i].capitalize(), variable = self.selected_head_fusion, value = self.head_types[i])
        attn_rollout.add_radiobutton(label = "All", variable = self.selected_head_fusion, value = "all")

        attn_opt.add_radiobutton(label = "Grad-CAM", variable = self.selected_head_fusion, value = "gradcam")
        self.raw_attn = BooleanVar()
        self.raw_attn.set(True)

        attn_opt.add_separator()

        attn_opt.add_checkbutton(label = "Raw attention", variable=self.raw_attn, onvalue=1, offvalue=0)

    
        attn_layer = Menu(menubar)
        self.selected_layer = IntVar()
        self.selected_layer.set(5)
        for i in range(len(self.model.module.pts_bbox_head.transformer.decoder.layers)):
            attn_layer.add_radiobutton(label = i, variable = self.selected_layer)
        attn_opt.add_cascade(label="Layer", menu=attn_layer)



        self.GT_bool, self.BB_bool, self.points_bool, self.scale, self.overlay, self.show_labels = BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar()

        add_opt = Menu(menubar)
        add_opt.add_checkbutton(label="Show GT Bounding Boxes", onvalue=1, offvalue=0, variable=self.GT_bool)
        add_opt.add_checkbutton(label="Show all Bounding Boxes", onvalue=1, offvalue=0, variable=self.BB_bool)
        add_opt.add_checkbutton(label="Show attention scale", onvalue=1, offvalue=0, variable=self.scale)
        add_opt.add_checkbutton(label="Overlay attention on image", onvalue=1, offvalue=0, variable=self.overlay)
        add_opt.add_checkbutton(label="Show predicted labels", onvalue=1, offvalue=0, variable=self.show_labels)

        menubar.add_cascade(
            label="Camera options",
            menu=camera_opt
        )

        menubar.add_cascade(
            label="Additional options",
            menu=add_opt
        )

        menubar.add_cascade(
            label="Attention options",
            menu=attn_opt
        )


        # self.head_fusion = "min"
        # self.discard_ratio = 0.9        
        # label1 = Label(self,text="Select a camera:")
        # label1.pack(fill=X, padx=5, pady=5)
        # frame = Frame(self)
        # frame.pack()
        
        # self.cameras = {'FRONT': 0, 'FRONT-RIGHT': 1, 'FRONT-LEFT': 2, 'BACK': 3, 'BACK-LEFT': 4, 'BACK-RIGHT': 5}
        # self.selected_camera = IntVar()
        
        # i = 0
        # for value,key in enumerate(self.cameras):
        #     Radiobutton(frame, text = key, variable = self.selected_camera, value = value).grid(row=0,column=i)
        #     i+=1




        plot_button = Button(self, command = self.visualize, text = "Visualize")
        
        plot_button.pack()


        
    def update_class(self, idx):
       self.text_label.set(f"Select bbox index: {class_names[self.labels[int(idx)].item()]} ({int(idx)})")
       self.BB_bool.set(0)
    
    def update_dr(self, idx):
        self.dr_text.set(f"Select discard ratio: {idx}")
 
    def update_thr(self, idx):
        self.thr_text.set(f"Select prediction threshold: {idx}")
        self.BB_bool.set(1)
        self.show_labels.set(1)
        
    def update_values(self):
        
        self.data = self.data_loader[self.data_idx.get()]
        if self.selected_head_fusion.get() != "gradcam":
            outputs = self.gen.get_all_attentions(self.data)
        else:
            outputs = self.gen.get_all_attentions(self.data, self.selected_bbox.get())

        self.nms_idxs = self.model.module.pts_bbox_head.bbox_coder.get_indexes() 
        self.outputs = outputs[0]["pts_bbox"]
        
        imgs = self.data["img"][0]._data[0].numpy()[0]
        imgs = imgs.transpose(0,2,3,1)[:,:900,:,:]
        self.imgs = imgs.astype(np.uint8)
 
        self.img_metas = self.data["img_metas"][0]._data[0][0]
        


    def visualize(self):
        
        fig = plt.figure(figsize=(40,20), layout="constrained")
        spec = fig.add_gridspec(3, 3)
        
        if self.old_data_idx != self.data_idx.get():
            self.old_data_idx = self.data_idx.get()
            self.selected_bbox.set(0)
            self.update_values()
                     
        if self.old_thr != self.selected_threshold.get() or self.old_data_idx != self.data_idx.get():
            self.old_thr = self.selected_threshold.get()
            self.thr_idxs = self.outputs['scores_3d'] > self.selected_threshold.get()
            self.selected_bbox.configure(to = len(self.thr_idxs.nonzero())-1)
            self.selected_bbox.set(0)
            self.labels = self.outputs['labels_3d'][self.thr_idxs]
            self.pred_bboxes = self.outputs["boxes_3d"][self.thr_idxs]
            self.pred_bboxes.tensor.detach()
            
        if self.old_layer_idx != self.attn_layer.get():
            self.old_layer_idx = self.attn_layer.get()
            self.gen.layer = self.attn_layer.get()
            

        if self.GT_bool.get():
            self.gt_bbox = self.gt_bboxes[self.data_idx.get()]
        else:
            self.gt_bbox = None

        # 0=CAMFRONT, 1=CAMFRONTRIGHT, 2=CAMFRONTLEFT, 3=CAMBACK, 4=CAMBACKLEFT, 5=CAMBACKRIGHT
        self.head_fusion = self.selected_head_fusion.get()
        self.discard_ratio = self.selected_discard_ratio.get()
        
        self.imgs_bbox = []
        for camidx in range(6):
            img = draw_lidar_bbox3d_on_img(
                    self.pred_bboxes if self.BB_bool.get() else self.pred_bboxes[self.selected_bbox.get()],
                    self.imgs[camidx],
                    self.img_metas['lidar2img'][camidx],
                    self.img_metas,
                    color=(0,255,0),
                    with_label = self.show_labels.get())  # BGR
            
            if self.gt_bbox:
                img = draw_lidar_bbox3d_on_img(
                        self.gt_bbox,
                        img,
                        self.img_metas['lidar2img'][camidx],
                        self.img_metas,
                        color=(255,0,0))
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs_bbox.append(img)
                
        cams = [2, 0, 1, 5, 3, 4]
        for i in range(6):
            if i < 3:
                ax = fig.add_subplot(spec[0, i]) 
            else:
                ax = fig.add_subplot(spec[2,i-3])
            ax.imshow(self.imgs_bbox[cams[i]])
            ax.axis('off')
            ax.set_title(f'{list(self.cameras.keys())[cams[i]]}')

        if self.head_fusion not in ("all", "gradcam"):
            attn = self.gen.generate_rollout(self.selected_bbox.get(), self.nms_idxs, self.selected_camera.get(), self.head_fusion, self.discard_ratio, self.raw_attn.get())
            attn = attn.view(29, 50).cpu().numpy()
            ax_attn = fig.add_subplot(spec[1,1])
            
            if self.overlay.get():
                attn = cv2.resize(attn, (self.imgs_bbox[0].shape[1], self.imgs_bbox[0].shape[0]))
                img = show_attn_on_img(self.imgs_bbox[self.selected_camera.get()], attn)
                attmap = ax_attn.imshow(img)
            
            else:
                attmap = ax_attn.imshow(attn)
            
            ax_attn.axis('off')
            ax_attn.set_title(f'{list(self.cameras.keys())[self.selected_camera.get()]}, layer {self.gen.layer}')
            
            if self.scale.get():  
                im_ratio = attn.shape[1]/attn.shape[0]
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                fig.colorbar(attmap, norm=norm, ax=ax_attn, orientation='horizontal', fraction=0.047*im_ratio)


                
            
        elif self.head_fusion == "gradcam":
            
            self.gen.get_all_attentions(self.data, self.selected_bbox.get())
            attn = self.gen.generate_attn_gradcam(self.selected_bbox.get(), self.nms_idxs, self.selected_camera.get())
            attn = attn.view(29, 50).cpu().numpy()
            ax_attn = fig.add_subplot(spec[1,1])
            attmap = ax_attn.imshow(attn)
            ax_attn.axis('off')
            ax_attn.set_title(f'{list(self.cameras.keys())[self.selected_camera.get()]}')  
            
            if self.scale.get():  
                im_ratio = attn.shape[1]/attn.shape[0]
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                fig.colorbar(attmap, norm=norm, ax=ax_attn, orientation='horizontal', fraction=0.047*im_ratio)
        
        elif self.head_fusion == "all":
            for i in range(len(self.head_types)):
                attn = self.gen.generate_rollout(self.selected_bbox.get(), self.nms_idxs, self.selected_camera.get(), self.head_types[i], self.discard_ratio, self.raw_attn.get())
                attn = attn.view(29, 50).cpu().numpy()
                ax_attn = fig.add_subplot(spec[1,i])
                attmap = ax_attn.imshow(attn)
                ax_attn.axis('off')
                ax_attn.set_title(f'{list(self.cameras.keys())[self.selected_camera.get()]} ({self.head_types[i]}), layer {self.gen.layer}')      
                
                if self.scale.get():  
                    im_ratio = attn.shape[1]/attn.shape[0]
                    norm = mpl.colors.Normalize(vmin=0, vmax=1)
                    fig.colorbar(attmap, norm=norm, ax=ax_attn, orientation='horizontal', fraction=0.047*im_ratio)
        
            
        if self.canvas: self.canvas.get_tk_widget().pack_forget()
        
        self.canvas = FigureCanvasTkAgg(fig, self)  
        self.canvas.draw()
    
        # placing the self.canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
        
        