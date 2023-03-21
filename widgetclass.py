import math

from PIL import Image
import requests
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import numpy as np


class AttentionVisualizer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.camidx = None
        self.indexes = 300
        self.url = ""
        self.cur_url = None
        self.pil_img = None
        self.tensor_img = None

        self.dec_attn_weights = None

        self.setup_widgets()

    def setup_widgets(self):
        self.sliders = [
            widgets.IntSlider(min=0, max=300,
                        step=1, description='Query index', value=150,
                        continuous_update=False,
                        layout=widgets.Layout(width='50%')
                        ),
            # # 0=CAMFRONT, 1=CAMFRONTRIGHT, 2=CAMFRONTLEFT, 3=CAMBACK, 4=CAMBACKLEFT, 5=CAMBACKRIGHT
            widgets.Dropdown(
                options=[('Front', 0), ('FrontRight', 1), ('FrontLeft', 2), ('Back', 3), ('BackLeft', 4), ('BackRight', 5)],
                value=0,
                description='Camera:',
            ),
             widgets.Text(
                value='http://images.cocodataset.org/val2017/000000039769.jpg',
                placeholder='Type something',
                description='URL (ENTER):',
                disabled=False,
                continuous_update=False,
                layout=widgets.Layout(width='100%')
            )           
        ]
        self.o = widgets.Output()

    def compute_features(self, data):
        model = self.model
        # use lists to store the outputs via up-values
        dec_self_attn_weights, dec_cross_attn_weights = [], []
        
        hooks = []
        for layer in model.module.pts_bbox_head.transformer.decoder.layers:
            hooks.append(
            layer.attentions[0].attn.register_forward_hook(
                lambda self, input, output: dec_self_attn_weights.append(output[1])
            ))
            hooks.append(
            layer.attentions[1].attn.register_forward_hook(
                lambda self, input, output: dec_cross_attn_weights.append(output[1])
            ))
        # propagate through the model
        data.pop("points")
        outputs = model(return_loss=False, rescale=True, **data)

        for hook in hooks:
            hook.remove()
        
        self.indexes = model.module.pts_bbox_head.bbox_coder.get_indexes()  



        self.dec_attn_weights = dec_cross_attn_weights[-1][self.camidx].cpu().min(axis=0)[0][self.indexes]
    
    def compute_on_image(self, data):

        self.compute_features(data)
    
    def update_chart(self, change):
        with self.o:
            clear_output()

            # j and i are the x and y coordinates of where to look at
            # sattn_dir is which direction to consider in the self-attention matrix
            # sattn_dot displays a red dot or not in the self-attention map
            queryidx, camidx, _ = [s.value for s in self.sliders]
            self.camidx = camidx
            fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(9, 4))
            
            self.compute_on_image(self.data)
            
            self.dec_attn_weights = self.dec_attn_weights[queryidx].detach()
            
            img = self.data["img"][0]._data[0].numpy()[0]
            img = img.transpose(0,2,3,1)
            img = img[self.camidx].astype(np.uint8)
            self.tensor_img = img

            axs[0].imshow(self.tensor_img)
            axs[0].axis('off')
            axs[0].set_title(f'Camidx: {self.camidx}')
            
            axs[1].imshow(self.dec_attn_weights.view(29,50), cmap='cividis', interpolation='nearest')
            axs[1].axis('off')
            axs[1].set_title(f'Queryid: {queryidx}')

            plt.show()
        
    def run(self):
      for s in self.sliders:
          s.observe(self.update_chart, 'value')
      self.update_chart(None)
      query, cam, url = self.sliders
      res = widgets.VBox(
      [
          url,
          widgets.HBox([query, cam]),
          self.o
      ])
      return res