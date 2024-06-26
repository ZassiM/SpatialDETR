import os
import torch
from mmcv import Config
from tkinter import filedialog as fd
import tomli
from mmcv.runner import (load_checkpoint, wrap_fp16_model)
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmcv.runner import init_dist
from mmdet.datasets import replace_ImageToTensor
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.cnn import xavier_init


class Model():
    ''' Model class. It manages the loading of the model, the weights and the dataset. '''

    def __init__(self):
        self.num_layers = 0
        self.use_mini_dataset = False

    def load_from_config(self, gpu_id=None):
        ''' Allows a fast loading if the model and dataset do not change.
            The moedl/dataset config file and weights can be defined inside config.toml. '''
        
        with open("/workspace/config.toml", mode="rb") as argsF:
            args = tomli.load(argsF)
            
        cfg_file = args["cfg_file"]
        weights_file = args["weights_file"]
        launcher = args["launcher"]
        if gpu_id is None:
            gpu_id = args["gpu_id"]

        self.load_model(cfg_file, weights_file, gpu_id, launcher)

    def load_model(self, cfg_file=None, weights_file=None, gpu_id=None, launcher="single"):
        ''' Loads the model with its weights and initialize all necessary objects. '''

        cfg_filetypes = (
            ('Config', '*.py'),
        )
        weights_filetypes = (
            ('Pickle', '*.pth'),
        )
        
        # Model config selection
        if cfg_file is None:
            cfg_file = fd.askopenfilename(
                title='Load model file',
                initialdir='/workspace/configs/submission/',
                filetypes=cfg_filetypes)
        
        # Model weights selection
        if cfg_file:
            if weights_file is None:
                weights_file = fd.askopenfilename(
                    title='Load weights',
                    initialdir='/workspace/work_dirs/checkpoints/',
                    filetypes=weights_filetypes)  
        
        if gpu_id is not None:
            self.gpu_id = gpu_id

        if not (cfg_file and weights_file):
            print("No file selected.")
            return
        
        # Model configuration needs to load weights
        args = {}
        args["config"] = cfg_file
        args["checkpoint"] = weights_file
        args["launcher"] = launcher

        # Model initialization 
        self.init_model(args)
        
        self.model_name = os.path.splitext(os.path.basename(cfg_file))[0]
        self.dataset_name = self.dataset.metadata['version']
        self.class_names = self.dataset.CLASSES
        self.num_layers = self.cfg.model.pts_bbox_head.transformer.decoder.num_layers
        self.num_heads = self.cfg.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[0].num_heads
        self.ori_shape = self.dataset[0]["img_metas"][0]._data["ori_shape"]
        self.pad_shape = self.dataset[0]["img_metas"][0]._data["pad_shape"]


        print("\nModel loaded.\n")

    def init_model(self, args):
        ''' Loads and initializes the model from a config file and loads the weights from a trained checkpoint '''

        cfg = Config.fromfile(args["config"])

        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        
        # import modules from plguin/xx, registry will be updated
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                else:
                    # import dir is the dirpath for the config file
                    _module_dir = os.path.dirname(args.config)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None
        # in case the test dataset is concatenated
        samples_per_gpu = 1
        
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        if args["launcher"] == 'single':
            distributed = False
        else:
            distributed = True
            init_dist(args["launcher"], **cfg.dist_params)

        dataset = build_dataset(cfg.data.val)

        ''' To use if necessary. '''
        # data_loader = build_dataloader(
        #     dataset,
        #     samples_per_gpu=samples_per_gpu,
        #     workers_per_gpu=cfg.data.workers_per_gpu,
        #     dist=distributed,
        #     shuffle=False)

        # Build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)

        checkpoint = load_checkpoint(model, args["checkpoint"], map_location='cpu')

        # Old versions did not save class info in checkpoints, this walkaround is for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        # Palette for visualization in segmentation tasks
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE

        ''' For Sanity Check. '''
        # model.pts_bbox_head.transformer.init_layers()
        # model.pts_bbox_head.transformer.init_weights()
        # model.img_backbone.init_weights()

        if not distributed:
            self.model = MMDataParallel(model, device_ids=[self.gpu_id])
        else:
            self.model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)

        self.dataset = dataset
        self.cfg = cfg





