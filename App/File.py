import tomli
import os
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import ttk
from mmcv.parallel import MMDataParallel

from App.Utils import random_data_idx, update_data_label
from XAI.Attention import Attention
from App.Model_init import init_app


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

    def update_progress_label(message):
        return f"{message}...{int(pb['value'])}%"

    def progress(i, message="Finished"):
        if pb['value'] + i < 100:
            pb['value'] += i
            value_label['text'] = update_progress_label(message)
            self.update()
        else:
            pb.destroy()
            value_label.destroy()
    
    pb = ttk.Progressbar(self, orient='horizontal', mode='determinate', length=280)
    pb.pack()

    value_label = ttk.Label(self, text=update_progress_label("Progress"))
    value_label.pack()
    #self.pack_slaves()

    progress(20, "Loading model with weights")
    # Model configuration needs to load weights
    args = {}
    args["config"] = cfg_file
    args["checkpoint"] = weights_file
    model, dataloader, checkpoint = init_app(args)
    progress(50, "Updating application")
            
    self.model = MMDataParallel(model, device_ids=[self.gpu_id.get()])
    self.data_loader = dataloader
    self.gen = Attention(self.model)
    self.new_model = True
    self.model_name = os.path.splitext(os.path.basename(cfg_file))[0]
    progress(30)

    if not self.started_app:
        self.start_app()
        random_data_idx(self)
        self.started_app = True

    update_data_label(self)
    print("Loading completed.")