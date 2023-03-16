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
from mmdet3d.core.visualizer import (show_multi_modality_result, show_result,
                                     show_seg_result)
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

    if mode == "test": dataset = build_dataset(cfg.data.test,default_args=dict(filter_empty_gt=False))
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
            
            try:
                example = dataset.prepare_train_data(i)
            except AttributeError:  # for Mono-3D datasets
                example = dataset.prepare_train_img(i)
                
                
            points = data.pop("points")
            #gt_bboxes = data.pop("gt_bboxes_3d")
            #gt_labels = data.pop("gt_labels_3d")
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            data["points"] = points
            
            
            # 0=CAMFRONT, 1=CAMFRONTRIGHT, 2=CAMFRONTLEFT, 3=CAMBACK, 4=CAMBACKLEFT, 5=CAMBACKRIGHT
            camidx = 2
            score_thr = 0.3
            
            inds = result[0]["pts_bbox"]['scores_3d'] > score_thr      
            
            gt_bboxes = dataset.get_ann_info(i)['gt_bboxes_3d']
            pred_bboxes = result[0]["pts_bbox"]["boxes_3d"][inds]
            
            img_metas = example['img_metas'][0]._data
            img_metas_test = data["img_metas"][0]._data[0][0]
            
            img = example['img'][0]._data.numpy()[camidx]
            img_test = data["img"][0]._data[0].numpy()[0][camidx]
            
            img = img.transpose(1, 2, 0)
            img_test = img_test.transpose(1,2,0)
            
            if gt_bboxes.tensor.shape[0] == 0:
                gt_bboxes = None
            
            filename = Path(img_metas['filename'][camidx]).name
            filename = filename.split('.')[0]
            
            show_multi_modality_result(
                img_test,
                gt_bboxes,
                pred_bboxes,
                img_metas_test['lidar2img'][camidx],
                args["show_dir"],
                filename,
                box_mode='lidar',
                img_metas=img_metas_test,
                gt_bbox_color = (0,0,255),
                pred_bbox_color = (0,255,0),
                show=True)
            
            #dataset.show(result, args["show_dir"], show = True, score_thr = 0.1)
        

            outputs.extend(result)
            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()


            #model.module.show_results_mod(data, results, out_dir = args["show_dir"], show = True)

                #camidx = 0
                
                # inds = results[0]['pts_bbox']['scores_3d'] > 0.5
                # pred_bbox = results[0]['pts_bbox']['boxes_3d'][inds]
                
                # labels=results[0]['pts_bbox']['labels_3d'][inds]
                # for l in labels:
                #     print(data_loader.dataset.CLASSES[l])

                # img_metas = data['img_metas'][0]._data[0][0]
                # # img = data['img'][0]._data[0][0]
                # # img = img[camidx].numpy().astype(np.uint8).transpose(1,2,0)

                # # no 3D gt bboxes, just show img
                # filename = Path(img_metas['filename'][camidx]).name
                # filename = filename.split('.')[0]
                
                # path = img_metas['filename'][camidx]
                # img = mmcv.imread(path)
                # img = mmcv.impad_to_multiple(img, 32, 0) #pad to (928,1600,3)
                # #mmcv.imshow(img)

                # show_multi_modality_result(
                #     img,
                #     None,
                #     pred_bbox,
                #     img_metas['lidar2img'][camidx],
                #     args["show_dir"],
                #     filename,
                #     box_mode='lidar',
                #     img_metas=img_metas,
                #     show=False)
                
                # mmcv.imshow(pred_img)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args["tmpdir"], args["gpu_collect"])     


    rank, _ = get_dist_info()
    if rank == 0:
        if args["out"]:
            print(f'\nwriting results to {args["out"]}')
            mmcv.dump(outputs, args["out"])
        #kwargs = {} if args["eval_options"] is None else args["eval_options"]
        kwargs = {}
        if args["format_only"]:
            dataset.format_results(outputs, **kwargs)
        if args["eval"]:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            #hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args["eval"], **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
