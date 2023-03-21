import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)

from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

import torch

from Explanation import Generator


class App(tk.Tk):

    def __init__(self, model, data, imgs, pred_bboxes, img_metas, thr_idxs, nms_idxs):
        super().__init__()

        # configure the root window
        self.title('My Awesome App')
        self.geometry('700x700')

        self.model = model
        self.data = data
        self.imgs = imgs
        self.pred_bboxes = pred_bboxes
        self.img_metas = img_metas
        self.thr_idxs = thr_idxs
        self.nms_idxs = nms_idxs
        
        self.gen = Generator(self.model)
        
        self.canvas = None

        label = ttk.Label(text="Select a camera:")
        label.pack(fill=tk.X, padx=5, pady=5)
        self.selected_camera = tk.IntVar()
        camera = ttk.Combobox(self, textvariable=self.selected_camera)

        # get first 3 letters of every month name
        camera['values'] = [0,1,2,3,4,5]
        # prevent typing a value
        camera['state'] = 'readonly'
        camera.pack(fill=tk.X, padx=5, pady=5)
                
        label1 = ttk.Label(text="Select bbox index:")
        label1.pack(fill=tk.X, padx=5, pady=5)
        self.selected_bbox = tk.IntVar()
        bboxes = ttk.Combobox(self, textvariable=self.selected_bbox)
        
        bboxes['values'] = self.thr_idxs.nonzero()[:,0].tolist()
        bboxes['state'] = 'readonly'
        bboxes.pack(fill=tk.X, padx=5, pady=5)
        
        plot_button = ttk.Button(master = self, 
                                command = self.plot,
                                text = "Plot")
        
        plot_button.pack()
        
        self.get_all_attentions()

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
            
        outputs = self.model(return_loss=False, rescale=True, **self.data)
        
        self.dec_self_attn_weights, self.dec_cross_attn_weights = dec_self_attn_weights, dec_cross_attn_weights
        
        for hook in hooks:
            hook.remove()
            
    def plot(self):
        
        if self.canvas: self.canvas.get_tk_widget().pack_forget()
        
        #fig = Figure(figsize = (20, 10),dpi = 100)
        fig = plt.figure(figsize=(22, 7), layout="constrained")
        
        # 0=CAMFRONT, 1=CAMFRONTRIGHT, 2=CAMFRONTLEFT, 3=CAMBACK, 4=CAMBACKLEFT, 5=CAMBACKRIGHT
        img_show = draw_lidar_bbox3d_on_img(
            self.pred_bboxes[self.selected_bbox.get()],
            self.imgs[self.selected_camera.get()],
            self.img_metas['lidar2img'][self.selected_camera.get()],
            self.img_metas,
            color=(255,0,0))

        spec = fig.add_gridspec(3, 3)
        # plotting the graph
        ax1= fig.add_subplot(spec[0, 0])
        ax1.imshow(img_show)
        ax1.axis('off')

        
        attn = self.gen.generate_rollout(self.dec_self_attn_weights, self.dec_cross_attn_weights, self.selected_bbox.get(), self.nms_idxs, self.selected_camera.get(), head_fusion = "min", discard_ratio = 0.9, raw = True)
        
        ax2 = fig.add_subplot(spec[0, 1])
        ax2.imshow(attn.view(29, 50).cpu())
        ax2.axis('off')

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