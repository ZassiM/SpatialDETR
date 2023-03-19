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
from ExplanationGenerator import Generator
import matplotlib.pyplot as plt

from vit_rollout import *
from mmdet3d.core.visualizer import (show_multi_modality_result,show_result,
                                     show_seg_result)
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from pathlib import Path
from mmcv import Config, DictAction, mkdir_or_exist, track_iter_progress




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

def build_data_cfg(config_path, skip_type, cfg_options):
    """Build data config for loading visualization data."""
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # extract inner dataset of `RepeatDataset` as `cfg.data.train`
    # so we don't need to worry about it later
    if cfg.data.train['type'] == 'RepeatDataset':
        cfg.data.train = cfg.data.train.dataset
    # use only first dataset for `ConcatDataset`
    if cfg.data.train['type'] == 'ConcatDataset':
        cfg.data.train = cfg.data.train.datasets[0]
    train_data_cfg = cfg.data.train
    # eval_pipeline purely consists of loading functions
    # use eval_pipeline for data loading
    train_data_cfg['pipeline'] = [
        x for x in cfg.eval_pipeline if x['type'] not in skip_type
    ]

    return cfg

      
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
        
        gen = Generator(model)
        for i, data in enumerate(data_loader):  
            if i<15: continue
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
            with torch.no_grad():
                points = data.pop("points")
                result = model(return_loss=False, rescale=True, **data)
                #data["points"] = points
                
            for hook in hooks:
                hook.remove()     

            # # 0=CAMFRONT, 1=CAMFRONTRIGHT, 2=CAMFRONTLEFT, 3=CAMBACK, 4=CAMBACKLEFT, 5=CAMBACKRIGHT

            indexes = model.module.pts_bbox_head.bbox_coder.get_indexes()  
            camidx = 0
            
            img = data["img"][0]._data[0].numpy()[0]
            img = img.transpose(0,2,3,1)
            img = img[camidx].astype(np.uint8)

            h, w = (29, 50)
            #dec_attn_weights = dec_cross_attn_weights[-1][camidx].min(axis=0)[0].cpu()
            dec_attn_weights = dec_cross_attn_weights
            
            full_attn = torch.eye(dec_attn_weights[0].size(-2), dec_attn_weights[0].size(-1))            
            
            for attn in dec_attn_weights: #6x(6x8x900x1450)
                attn = attn[camidx].cpu() #8x900x1450
                attn = attn.min(axis=0)[0] #900x1450

                flat = attn.view(attn.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*0.5), -1, False)
                indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attn.size(-2),attn.size(-1))
                a = (attn + 1.0*I)/2
                a = a / a.sum(dim=-2)

                full_attn = attn
                
                
            
            #dec_attn_weights = dec_attn_weights[indexes]
            dec_attn_weights = full_attn[indexes]
            inds = result[0]["pts_bbox"]['scores_3d'] > 0.6
            pred_bboxes = result[0]["pts_bbox"]["boxes_3d"][inds]
            img_metas = data["img_metas"][0]._data[0][0]
            
            fig, axs = plt.subplots(ncols=5, nrows=2, figsize=(22, 7))
            k=0
            for idx, ax_i in zip(inds.nonzero(), axs.T):
                
                ax = ax_i[0]
                mask = dec_attn_weights[idx].view(h, w)
                ax.imshow(mask)
                ax.axis('off')
                ax.set_title(f'query id: {indexes[k]}')
                
                ax = ax_i[1]
                img_show = draw_lidar_bbox3d_on_img(
                    pred_bboxes[idx],
                    img,
                    img_metas['lidar2img'][camidx],
                    img_metas,
                    color=(255,0,0))
                #ax.imshow(img_show)
                mmcv.imshow(img_show, win_name='attn', wait_time=0)
                ax.axis('off')
                class_name = class_names[result[0]['pts_bbox']['labels_3d'][k]]
                score = result[0]['pts_bbox']['scores_3d'][k].item()
                score = round(score,3)
                
                ax.set_title(f'{class_name}: {score}%')
                k+=1

            fig.tight_layout()           
            
            debug = 1

            
            # 
            # mmcv.imshow(mask, "Attn")
            
                    
            # outputs.extend(result)
            # batch_size = len(result)
            # for _ in range(batch_size):
            #     prog_bar.update()
            
            

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args["tmpdir"], args["gpu_collect"])     



if __name__ == '__main__':
    main()
