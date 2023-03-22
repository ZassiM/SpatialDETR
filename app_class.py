from tkinter import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

import numpy as np
import cv2


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

class App(Tk):

    def __init__(self, model, data_loader):
        super().__init__()
        
        #style = Style(self)
        
        self.title('Attention Visualization')
        self.geometry('1500x1500')

        self.model = model
        self.data_loader = data_loader
        
        
        self.gen = Generator(self.model)
        self.canvas = None
        self.thr_idxs = []
        self.old_data_idx = None

        label0 = Label(text="Select data index:", anchor = CENTER)
        label0.pack(fill=X, padx=5, pady=5)
        self.data_idx = Scale(self, from_=0, to=len(self.data_loader) - 1, orient=HORIZONTAL)
        self.data_idx.set(0)
        self.data_idx.pack()
        

        label1 = Label(text="Select a camera:")
        label1.pack(fill=X, padx=5, pady=5)
        
        self.cameras = {'FRONT': 0, 'FRONT-RIGHT': 1, 'FRONT-LEFT': 2, 'BACK': 3, 'BACK-LEFT': 4, 'BACK-RIGHT': 5}
        self.selected_camera = IntVar()
        i = 0
        frame = Frame(self)
        frame.pack()
        for value,key in enumerate(self.cameras):
            Radiobutton(frame, text = key, variable = self.selected_camera, value = value).grid(column=i, row=0)
            #radiobutton.pack(anchor = CENTER)
            i+=1
            
        self.text_label = StringVar()
        self.text_label.set("Select bbox index:")
        label2 = Label(textvariable = self.text_label)
        label2.pack(fill=X, padx=5, pady=5)
        

        self.selected_bbox = Scale(self, from_=0, to=len(self.thr_idxs), orient=HORIZONTAL)
        self.selected_bbox.set(0)
        self.selected_bbox.pack()
        
        
        plot_button = Button(master = self, 
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
        
        
        # for i,data in enumerate(self.data_loader):
        #     if i == int(self.data_idx.get()): 
        #         self.data = data
        #         self.data.pop("points")
        #         self.old_data_idx = self.data_idx.get()
        #         break
        
        self.data = self.data_loader[self.data_idx.get()]
        self.data.pop("points")
        self.old_data_idx = self.data_idx.get()
            
        imgs = self.data["img"][0]._data[0].numpy()[0]
        imgs = imgs.transpose(0,2,3,1)
        self.imgs = imgs.astype(np.uint8)
        
        outputs = self.model(return_loss=False, rescale=True, **self.data)
        
        for hook in hooks:
            hook.remove()
            
        self.dec_self_attn_weights, self.dec_cross_attn_weights = dec_self_attn_weights, dec_cross_attn_weights
        self.nms_idxs = self.model.module.pts_bbox_head.bbox_coder.get_indexes()  

        
        self.thr_idxs = outputs[0]["pts_bbox"]['scores_3d'] > 0.6
        self.selected_bbox.configure(to = len(self.thr_idxs.nonzero())-1)

        #self.bboxes['values'] = self.thr_idxs.nonzero()[:,0].tolist()
        self.labels = outputs[0]["pts_bbox"]['labels_3d']
        #self.text_label.set(f"Bboxes: {class_names[self.thr_idxs]}")
        self.pred_bboxes = outputs[0]["pts_bbox"]["boxes_3d"][self.thr_idxs]
        self.img_metas = self.data["img_metas"][0]._data[0][0]
        
        self.pred_bboxes.tensor.detach()


    def visualize(self):
        if self.old_data_idx != self.data_idx.get():
            self.get_all_attentions()
        
        if self.canvas: self.canvas.get_tk_widget().pack_forget()
        
        #fig = Figure(figsize = (20, 10),dpi = 100)
        fig = plt.figure(figsize=(40,20), layout="constrained")
        
        # 0=CAMFRONT, 1=CAMFRONTRIGHT, 2=CAMFRONTLEFT, 3=CAMBACK, 4=CAMBACKLEFT, 5=CAMBACKRIGHT
        self.camidx = self.selected_camera.get()
        
        imgs = []
        for camidx in range(6):
            img = draw_lidar_bbox3d_on_img(
                    self.pred_bboxes[self.selected_bbox.get()],
                    self.imgs[camidx],
                    self.img_metas['lidar2img'][camidx],
                    self.img_metas,
                    color=(255,0,0))
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    
        
        attn = self.gen.generate_rollout_app(self.dec_self_attn_weights, self.dec_cross_attn_weights, self.selected_bbox.get(), self.nms_idxs, self.camidx, head_fusion = "min", discard_ratio = 0.9, raw = True)
        ax_attn = fig.add_subplot(spec[1,1])
        ax_attn.imshow(attn.view(29, 50).cpu())
        ax_attn.axis('off')

        #fig.subplots_adjust(wspace=0, hspace=0)
        #fig.tight_layout()
        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(fig,
                                master = self)  
        self.canvas.draw()
    
        # placing the self.canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
        
        