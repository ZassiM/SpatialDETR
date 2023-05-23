''' Functions for loading the model. '''
import tomli
import os
from tkinter import filedialog as fd
from mmcv.parallel import MMDataParallel

from App.Utils import random_data_idx, update_info_label
from Explainability.Attention import Attention
from App.Model import init_app


def load_from_config(self):
    with open("config.toml", mode="rb") as argsF:
        args = tomli.load(argsF)
        
    cfg_file = args["cfg_file"]
    weights_file = args["weights_file"]
    gpu_id = args["gpu_id"]

    load_model(self, cfg_file, weights_file, gpu_id)


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
    self.dataloader = dataloader
    self.Attention = Attention(self.model)
    self.model_name = os.path.splitext(os.path.basename(cfg_file))[0]
    self.dataloader_name = self.dataloader.dataset.metadata['version']
    self.class_names = self.dataloader.dataset.CLASSES
    print("Loading completed.\n")
    
    self.new_model = True

    if not self.started_app:
        self.start_app()
        self.started_app = True
        random_data_idx(self)

    update_info_label(self)

