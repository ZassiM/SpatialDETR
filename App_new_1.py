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
        
        
        self.GT_bool, self.BB_bool, self.points_bool, self.scale, self.overlay, self.show_labels = BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar()

        menubar = Menu(self)
        self.config(menu=menubar)

        # create the file_menu
        file_menu = Menu(
            menubar,
            tearoff=0
        )

        # add menu items to the File menu
        file_menu.add_command(label='New')
        file_menu.add_command(label='Open...')
        file_menu.add_command(label='Close')
        file_menu.add_separator()

        # add Exit menu item
        file_menu.add_command(
            label='Exit',
            command=self.destroy
        )

        # add the File menu to the menubar
        menubar.add_cascade(
            label="File",
            menu=file_menu
        )
        # create the Help menu
        help_menu = Menu(
            menubar,
            tearoff=0
        )

        help_menu.add_command(label='Welcome')
        help_menu.add_command(label='About...')

        # add the Help menu to the menubar
        menubar.add_cascade(
            label="Help",
            menu=help_menu
        )


        
    def update_class(self, idx):
       self.text_label.set(f"Select bbox index: {class_names[self.labels[int(idx)].item()]} ({int(idx)})")
       self.BB_bool.set(0)
    
    def update_dr(self, idx):
        self.dr_text.set(f"Select discard ratio: {idx}")

    def update_layer(self, idx):
        self.layer_text.set(f"Select layer to visualize: {idx}")
        
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
        
        