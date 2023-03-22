import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

import torch
import numpy as np


from Explanation import Generator

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

class App(tk.Tk):

    def __init__(self, model, data_loader):
        super().__init__()
        
        style = ttk.Style(self)
        style.theme_use('clam')
        
        self.title('Attention Visualization')
        self.geometry('1500x1500')

        self.model = model
        self.data_loader = data_loader
        
        
        self.gen = Generator(self.model)
        self.canvas = None
        self.thr_idxs = []
        self.old_data_idx = None

        label0 = ttk.Label(text="Select data index:")
        label0.pack(fill=tk.X, padx=5, pady=5)
        self.data_idx = tk.Scale(self, from_=0, to=len(self.data_loader), orient=tk.HORIZONTAL)
        self.data_idx.set(0)
        self.data_idx.pack()
        

        label1 = ttk.Label(text="Select a camera:")
        label1.pack(fill=tk.X, padx=5, pady=5)
        self.selected_camera = tk.StringVar()
        camera = ttk.Combobox(self, textvariable=self.selected_camera)
        # 0=CAMFRONT, 1=CAMFRONTRIGHT, 2=CAMFRONTLEFT, 3=CAMBACK, 4=CAMBACKLEFT, 5=CAMBACKRIGHT

        self.cameras = {'FRONT': 0, 'FRONT-RIGHT': 1, 'FRONT-LEFT': 2, 'BACK': 3, 'BACK-LEFT': 4, 'BACK-RIGHT': 5}

        camera['values'] = list(self.cameras.keys())
        camera['state'] = 'readonly'
        camera.current(0)
        camera.pack(fill=tk.X, padx=5, pady=5)

        self.text_label = tk.StringVar()
        self.text_label.set("Select bbox index:")
        label2 = ttk.Label(textvariable = self.text_label)
        
        label2.pack(fill=tk.X, padx=5, pady=5)
        self.selected_bbox = tk.IntVar()
        self.bboxes = ttk.Combobox(self, textvariable=self.selected_bbox)
        
        self.bboxes['values'] = self.thr_idxs
        self.bboxes['state'] = 'readonly'
        self.bboxes.pack(fill=tk.X, padx=5, pady=5)
        
        plot_button = ttk.Button(master = self, 
                                command = self.visualize,
                                text = "Visualize")
        
        plot_button.pack()
        

    def get_all_attentions(self):
         
        dec_self_attn_weights, dec_cross_attn_weights = [], []
        
        hooks = []
        for layer in self.model.module.pts_bbox_head.transformer.decoder.layers:
            hooks.append(
            layer.attentions[0].attn.register_forward_hook(
                lambda self, input, output: dec_self_attn_weights.append(output[1])
            ))
            hooks.append(
            layer.attentions[1].attn.register_forward_hook(
                lambda self, input, output: dec_cross_attn_weights.append(output[1])
            ))
        
        if self.old_data_idx != self.data_idx.get():
            for i,data in enumerate(self.data_loader):
                if i == int(self.data_idx.get()): 
                    self.data = data
                    self.data.pop("points")
                    self.old_data_idx = self.data_idx.get()
                    break
        
            
        imgs = self.data["img"][0]._data[0].numpy()[0]
        imgs = imgs.transpose(0,2,3,1)
        self.imgs = imgs.astype(np.uint8)
        
        outputs = self.model(return_loss=False, rescale=True, **self.data)
        
        for hook in hooks:
            hook.remove()
            
        self.dec_self_attn_weights, self.dec_cross_attn_weights = dec_self_attn_weights, dec_cross_attn_weights
        self.nms_idxs = self.model.module.pts_bbox_head.bbox_coder.get_indexes()  

        
        self.thr_idxs = outputs[0]["pts_bbox"]['scores_3d'] > 0.6
        self.bboxes['values'] = self.thr_idxs.nonzero()[:,0].tolist()
        self.text_label.set(f"Bboxes: {class_names[self.thr_idxs]}")
        self.pred_bboxes = outputs[0]["pts_bbox"]["boxes_3d"][self.thr_idxs]
        self.img_metas = self.data["img_metas"][0]._data[0][0]
        
        self.pred_bboxes.tensor.detach()


    def visualize(self):
        
        self.get_all_attentions()
        
        if self.canvas: self.canvas.get_tk_widget().pack_forget()
        
        #fig = Figure(figsize = (20, 10),dpi = 100)
        fig = plt.figure(figsize=(40,30), layout="constrained")
        
        # 0=CAMFRONT, 1=CAMFRONTRIGHT, 2=CAMFRONTLEFT, 3=CAMBACK, 4=CAMBACKLEFT, 5=CAMBACKRIGHT
        self.camidx = self.cameras[self.selected_camera.get()]
        
        imgs = []
        for camidx in range(6):
            img = draw_lidar_bbox3d_on_img(
            self.pred_bboxes[self.selected_bbox.get()],
            self.imgs[camidx],
            self.img_metas['lidar2img'][camidx],
            self.img_metas,
            color=(255,0,0))
            imgs.append(img)

        cams = [2, 0, 1, 5, 3, 4]
        
        spec = fig.add_gridspec(3, 3)
        
        for i in range(6):
            if i < 3:
                ax = fig.add_subplot(spec[0, i]) 
            else:
                ax = fig.add_subplot(spec[2,i-3])
            
            ax.imshow(imgs[cams[i]])
            ax.axis('off')
            ax.set_title(f'{list(self.cameras.keys())[cams[i]]}')
    
        
        attn = self.gen.generate_rollout(self.dec_self_attn_weights, self.dec_cross_attn_weights, self.selected_bbox.get(), self.nms_idxs, self.camidx, head_fusion = "min", discard_ratio = 0.9, raw = True)
        
        ax_attn = fig.add_subplot(spec[1,1])
        ax_attn.imshow(attn.view(29, 50).cpu())
        ax_attn.axis('off')

        fig.tight_layout()
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(fig,
                                master = self)  
        self.canvas.draw()
    
        # placing the self.canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
        
        
        
        
        



if __name__ == "__main__":
  app = App()
  app.mainloop()