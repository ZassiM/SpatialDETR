from modules.Model import Model
from modules.Explainability import ExplainableTransformer
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np
import mmcv
import warnings
import time
import os
import gc


def main():
    warnings.filterwarnings("ignore")

    config = {
        "mechanism": "Gradient Rollout",     #  'Raw Attention', 'Grad-CAM', 'Gradient Rollout', 'Random'
        "perturbation_type": "negative",  #  'positive', 'negative'
        "pred_threshold": 0.4,
        "discard_threshold": 0.3,
        "head_fusion_method": "zero_clamp_mean",      #  'max', 'min', 'mean'
        "layer_fusion_method": "max",    #  'max', 'min', 'mean', 'last'
        "maps_quality": "High",
        "remove_pad": True,
        "grad_rollout_handle_residual": True,
        "grad_rollout_apply_rule": True,
    }
    
    perturbation_steps = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]

    model = Model()
    model.load_from_config(gpu_id=0)

    xai_generator = ExplainableTransformer(model)

    # Info Text
    info = "*" * (38 + len(config['mechanism']))
    info += (f"\nEvaluating {config['mechanism']} with {config['perturbation_type']} perturbation\n")
    info += "*" * (38 + len(config['mechanism']))
    info += "\n"
    for k, v in config.items():
        info += f"{k}: {v}\n"
    info += "*" * (38 + len(config['mechanism']))

    eval_folder = f"eval_results/{config['mechanism']}"
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    file_name = f"{model.model_name}_{model.dataset_name}"
    file_path = os.path.join(eval_folder, file_name)
    counter = 1
    while os.path.exists(file_path+".txt"):
        file_name_new = f"{file_name}_{counter}"
        file_path = os.path.join(eval_folder, file_name_new)
        counter += 1

    file_path += ".txt"
    with open(file_path, "a") as file:
        file.write(f"{info}\n")

    print(info)
    start_time = time.time()
    for i in range(len(perturbation_steps)):
        print(f"\nNumber of tokens removed: {perturbation_steps[i] * 100} %")
        evaluate_step(
            model=model,
            xai_generator=xai_generator,
            step=perturbation_steps[i],
            eval_file=file_path,
            config=config)
        gc.collect()
        torch.cuda.empty_cache()
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Completed (elapsed time {total_time} seconds).\n")

    with open(file_path, "a") as file:
        file.write("--------------------------\n")
        file.write(f"Elapsed time: {total_time}\n")


def evaluate_step(model, xai_generator, step, eval_file, config):
    # general setting
    expl_type = config["mechanism"]
    discard_threshold = config["discard_threshold"]
    pred_threshold = config["pred_threshold"]
    perturbation_type = config["perturbation_type"]
    maps_quality = config["maps_quality"]
    remove_pad = config["remove_pad"]

    # transformer fusion
    layer_fusion_method = config["layer_fusion_method"]
    head_fusion_method = config["head_fusion_method"]

    # for gradient rollout
    handle_residual = config["grad_rollout_handle_residual"]
    apply_rule = config["grad_rollout_apply_rule"]
    
    dataset = model.dataset
    evaluation_lenght = len(dataset)
    outputs_pert = []

    model.model.eval()

    prog_bar = mmcv.ProgressBar(evaluation_lenght)

    for dataidx in range(evaluation_lenght):
        data = dataset[dataidx]
        metas = [[data['img_metas'][0].data]]
        img = [data['img'][0].data.unsqueeze(0)]  # img[0] = torch.Size([1, 6, 3, 928, 1600])
        data['img_metas'][0] = DC(metas, cpu_only=True)
        data['img'][0] = DC(img)
    
        if "points" in data.keys():
            data.pop("points")

        if expl_type != "Random":
            # Attention scores are extracted, together with gradients if grad-CAM is selected
            output_og = xai_generator.extract_attentions(data)

            # Extract predicted bboxes and their labels
            outputs = output_og[0]["pts_bbox"]
            nms_idxs = model.model.module.pts_bbox_head.bbox_coder.get_indexes().cpu()
            thr_idxs = outputs['scores_3d'] > pred_threshold
            labels = outputs['labels_3d'][thr_idxs]
            
            bbox_idx = list(range(len(labels)))
            if expl_type in ["Grad-CAM", "Gradient Rollout"]:
                xai_generator.extract_attentions(data, bbox_idx)

            if len(bbox_idx) > 0:
                xai_generator.generate_explainability(
                    expl_type=expl_type,
                    head_fusion=head_fusion_method,
                    handle_residual=handle_residual,
                    apply_rule=apply_rule)
                xai_generator.select_explainability(
                    nms_idxs=nms_idxs,
                    bbox_idx=bbox_idx,
                    discard_threshold=discard_threshold,
                    maps_quality=maps_quality,
                    remove_pad=remove_pad,
                    layer_fusion_method=layer_fusion_method)
                xai_maps = xai_generator.xai_maps
            else:
                print("\nNO DETECTION - GENERATING RANDOM XAI MAP")
                xai_maps = torch.rand(6, model.ori_shape[0], model.ori_shape[1])
        
        else:
            # TO FIX!!!
            if remove_pad:
                xai_maps = torch.rand(6, model.ori_shape[0], model.ori_shape[1])
            else:
                xai_maps = torch.rand(6, model.pad_shape[0], model.pad_shape[1])

        # Perturbate the input image with the XAI maps
        img = img[0][0]
        if remove_pad:
            img = img[:, :, :model.ori_shape[0], :model.ori_shape[1]]  # [num_cams x height x width x channels]

        # defect data: 475
        img_pert_list = []  # list of perturbed images
        for cam in range(len(xai_maps)):
            img_pert = img[cam].permute(1, 2, 0).numpy()

            xai_cam = xai_maps[cam]

            if perturbation_type == "negative":
                xai_cam = -xai_cam
            elif perturbation_type == "positive":
                xai_cam = xai_cam
            else:
                raise NotImplementedError

            num_pixels_removed = int(step * xai_cam.numel())
            if dataidx == 0:
                print("Number of Pixel Removed for Cam {1}: {0}".format(num_pixels_removed, cam))
            _, indices = torch.topk(xai_cam.flatten(), num_pixels_removed)

            row_indices, col_indices = indices // xai_cam.size(1), indices % xai_cam.size(1)
            img_pert[row_indices, col_indices] = np.mean(img_pert, axis=(0, 1))
            img_pert_list.append(img_pert)
        
        # if there are detected objects, apply the perturbated images to the data input
        if len(img_pert_list) > 0:
            # save_img the perturbed 6 camera images into the data input
            img_pert_list = torch.from_numpy(np.stack(img_pert_list))
            img = [img_pert_list.permute(0, 3, 1, 2).unsqueeze(0)]
            data['img'][0] = DC(img)

        # Apply perturbated images to the model
        with torch.no_grad():
            output_pert = model.model(return_loss=False, rescale=True, **data)

        outputs_pert.extend(output_pert)
        prog_bar.update()
    
    kwargs = {}
    eval_kwargs = model.cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="bbox", **kwargs))
    eval_results = dataset.evaluate(outputs_pert, **eval_kwargs)
    mAP = eval_results['pts_bbox_NuScenes/mAP']
    mATE = eval_results['pts_bbox_NuScenes/mATE']
    mASE = eval_results['pts_bbox_NuScenes/mASE']
    mAOE = eval_results['pts_bbox_NuScenes/mAOE']
    mAVE = eval_results['pts_bbox_NuScenes/mAVE']
    mAAE = eval_results['pts_bbox_NuScenes/mAAE']
    NDScore = eval_results['pts_bbox_NuScenes/NDS']

    with open(eval_file, "a") as file:
        file.write(f"Number of tokens: {int(step*100)} %\n")
        file.write(f"mAP: {mAP}\n")
        file.write(f"mATE: {mATE}\n")
        file.write(f"mASE: {mASE}\n")
        file.write(f"mAOE: {mAOE}\n")
        file.write(f"mAVE: {mAVE}\n")
        file.write(f"mAAE: {mAAE}\n")
        file.write(f"NDS: {NDScore}\n")
        file.write("--------------------------\n")
    
    print(f"File {eval_file} updated.\n")
    gc.collect()
    del outputs_pert


if __name__ == '__main__':
    main()
