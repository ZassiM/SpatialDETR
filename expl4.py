# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import os
import tomli
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

from mmdetection3d.tools.misc.browse_dataset import show_proj_bbox_img

from PIL import Image
import numpy as np
import cv2
from Explanation import Generator
import matplotlib.pyplot as plt

from vit_rollout import *
from mmdet3d.core.visualizer import (show_multi_modality_result,show_result,
                                     show_seg_result)
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from pathlib import Path
from mmcv import Config, DictAction, mkdir_or_exist, track_iter_progress

from matplotlib.widgets import Slider

import matplotlib.gridspec as gridspec



def init(args):
    
    args["config"] = args["config_"+args["model"]]
    args["checkpoint"] = args["checkpoint_"+args["model"]]

    if args["out"] is not None and not args["out"].endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    
    cfg = Config.fromfile(args["config"])
    #cfg = build_data_cfg(args["config"], args["skip_type"], cfg_options = None)
    
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
                
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

    gpu_ids = [args["gpu_id"]]

    # init distributed env first, since logger depends on the dist info.
    if args["launcher"] == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args["launcher"], **cfg.dist_params)
    
    # build the dataloader
    mode = args["mode"]

    if mode == "test": dataset = build_dataset(cfg.data.test)
    elif mode == "train": dataset = build_dataset(cfg.data.train)
    elif mode == "val": dataset = build_dataset(cfg.data.val)
    else: raise ValueError(f"{mode} is not a valid mode.\n")
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args["checkpoint"], map_location='cpu')
    if args["fuse_conv_bn"]:
        model = fuse_conv_bn(model)

    # old versions did not save class info in checkpoints, this walkaround is for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    
    return model, dataset, data_loader, gpu_ids, cfg, distributed

      
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    mask = heatmap + np.float32(img)
    mask = mask / np.max(mask)
    return np.uint8(255 * mask)

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


def main():
    
    
    with open("args.toml", mode = "rb") as argsF:
        args = tomli.load(argsF)
    
    model, dataset, data_loader, gpu_ids, cfg, distributed = init(args)

    
    if not distributed:
        model = MMDataParallel(model, device_ids = gpu_ids)

        model.eval()
        outputs = []
        prog_bar = mmcv.ProgressBar(len(dataset))
        
        for i, data in enumerate(data_loader):  
            if i<29: continue

            # # 0=CAMFRONT, 1=CAMFRONTRIGHT, 2=CAMFRONTLEFT, 3=CAMBACK, 4=CAMBACKLEFT, 5=CAMBACKRIGHT
            camtarget = 0
            
            data.pop("points")
            result = model(return_loss=False, rescale=True, **data)
                
            indexes = model.module.pts_bbox_head.bbox_coder.get_indexes()  

            img = data["img"][0]._data[0].numpy()[0]
            img = img.transpose(0,2,3,1)
            img = img[camtarget].astype(np.uint8)
            
            inds = result[0]["pts_bbox"]['scores_3d'] > 0.6
            pred_bboxes = result[0]["pts_bbox"]["boxes_3d"][inds]
            img_metas = data["img_metas"][0]._data[0][0]
            
            pred_bboxes.tensor.detach()
            
            h, w = (29, 50)
            gen = Generator(model)
            
            fig = plt.figure(figsize=(22, 7), layout="constrained")
            #nbboxes = len(inds.nonzero())
            nbboxes = 4
            spec = fig.add_gridspec(3, 2*nbboxes)

            k = 0
            for target in inds.nonzero():
                if k == 2*nbboxes: break
                #if result[0]['pts_bbox']['labels_3d'][target] in (0,8): continue
                aximg = fig.add_subplot(spec[0, k:k+2])
                img_show = draw_lidar_bbox3d_on_img(
                    pred_bboxes[target],
                    img,
                    img_metas['lidar2img'][camtarget],
                    img_metas,
                    color=(255,0,0))
                img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
                aximg.imshow(img_show)
                aximg.axis('off')
                class_name = class_names[result[0]['pts_bbox']['labels_3d'][target]]
                score = result[0]['pts_bbox']['scores_3d'][target].item()
                score = round(score,3)
                
                aximg.set_title(f'{class_name}: {score}%')
                

                #attn0 = gen.generate_ours(data, target, indexes, camtarget)
                attn0 = gen.generate_rollout(data, target, indexes, camtarget, head_fusion = "max", discard_ratio = 0.9)
                attn1 = gen.generate_rollout(data, target, indexes, camtarget, head_fusion = "min", discard_ratio = 0.9)
                attn2 = gen.generate_rollout(data, target, indexes, camtarget, head_fusion = "min", discard_ratio = 0.9, raw = True)
                attn3 = gen.generate_attn_gradcam(data, target, indexes, camtarget)
                
                ax_attn0 = fig.add_subplot(spec[1, k:k+1])
                ax_attn1 = fig.add_subplot(spec[1, k+1:k+2])
                ax_attn2 = fig.add_subplot(spec[2, k:k+1])
                ax_attn3 = fig.add_subplot(spec[2, k+1:k+2])
                #attn = (attn - attn.min()) / (attn.max() - attn.min())
                
                ax_attn0.imshow(attn0.view(h, w).cpu())
                ax_attn0.axis('off')
                ax_attn0.set_title('Rollout with max fusion')
                
                ax_attn1.imshow(attn1.view(h, w).cpu())
                ax_attn1.axis('off')
                ax_attn1.set_title('Rollout with min fusion')
                
                ax_attn2.imshow(attn2.view(h, w).cpu())
                ax_attn2.axis('off')
                ax_attn2.set_title('Raw attention')
                
                ax_attn3.imshow(attn3.view(h, w).cpu())
                ax_attn3.axis('off')
                ax_attn3.set_title('Grad-CAM')

                k+=2

            fig.tight_layout()    
            plt.show()
            debug = 1
            plt.close(fig)
            
            
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args["tmpdir"], args["gpu_collect"])     


if __name__ == '__main__':
    main()



