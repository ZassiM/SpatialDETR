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

    expl_types = ["Attention Rollout", "Grad-CAM", "Gradient Rollout", "RandExpl"]
    ObjectDetector = Model()
    ObjectDetector.load_from_config()
    ExplainabiliyGenerator = ExplainableTransformer(ObjectDetector)

    evaluate(ObjectDetector, ExplainabiliyGenerator, expl_types[0], negative_pert=False)

def evaluate(Model, ExplGen, expl_type, negative_pert=False):
    txt_del = "*" * (38 + len(expl_type))
    info = txt_del
    if not negative_pert:
        info += (f"\nEvaluating {expl_type} with positive perturbation\n")
    else:
        info += (f"\nEvaluating {expl_type} with negative perturbation\n")
    info += txt_del

    eval_folder = f"eval_results/{expl_type}"
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    file_name = f"{Model.model_name}_{Model.dataloader_name}"
    file_path = os.path.join(eval_folder, file_name)
    counter = 1
    while os.path.exists(file_path+".txt"):
        file_name_new = f"{file_name}_{counter}"
        file_path = os.path.join(eval_folder, file_name_new)
        counter += 1

    file_path += ".txt"
    with open(file_path, "a") as file:
        file.write(f"{info}\n")

    base_size = 29 * 50
    pert_steps = [0, 0.25, 0.5, 0.75, 1]

    print(info)
    start_time = time.time()
    for step in range(len(pert_steps)):
        num_tokens = int(base_size * pert_steps[step])
        print(f"\nNumber of tokens removed: {num_tokens} ({pert_steps[step] * 100} %)")
        evaluate_step(Model, ExplGen, expl_type, num_tokens=num_tokens, negative_pert=negative_pert, eval_file=file_path, remove_pad=False)
        gc.collect()
        torch.cuda.empty_cache()
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Completed (elapsed time {total_time} seconds).\n")

    with open(file_path, "a") as file:
        file.write("--------------------------\n")
        file.write(f"Elapsed time: {total_time}\n")

def evaluate_step(Model, ExplGen, expl_type, num_tokens, eval_file, negative_pert=False, remove_pad=False):
    pred_threshold = 0.5
    #layer = Model.num_layers - 1
    layer = 0
    head_fusion, discard_ratio, raw_attention, handle_residual, apply_rule = \
        "max", 0.5, True, True, True
    dataset = Model.dataset
    evaluation_lenght = len(dataset)
    outputs_pert = []

    prog_bar = mmcv.ProgressBar(evaluation_lenght)

    for dataidx in range(evaluation_lenght):

        data = dataset[dataidx]
        metas = [[data['img_metas'][0].data]]
        img = [data['img'][0].data.unsqueeze(0)]  # img[0] = torch.Size([1, 6, 3, 928, 1600])
        data['img_metas'][0] = DC(metas, cpu_only=True)
        data['img'][0] = DC(img)
    
        if "points" in data.keys():
            data.pop("points")

        # Attention scores are extracted, together with gradients if grad-CAM is selected
        output_og = ExplGen.extract_attentions(data)

        # Extract predicted bboxes and their labels
        outputs = output_og[0]["pts_bbox"]
        del output_og
        torch.cuda.empty_cache()
        thr_idxs = outputs['scores_3d'] > pred_threshold
        nms_idxs = Model.model.module.pts_bbox_head.bbox_coder.get_indexes().cpu()
        labels = outputs['labels_3d'][thr_idxs]
        
        bbox_idx = list(range(len(labels)))
        if expl_type in ["Grad-CAM", "Gradient Rollout"]:
            ExplGen.extract_attentions(data, bbox_idx)

        ExplGen.generate_explainability(expl_type, bbox_idx, nms_idxs, head_fusion, discard_ratio, raw_attention, handle_residual, apply_rule, remove_pad)
        attn_list = ExplGen.attn_list[layer]

        # Perturbate the input image with the XAI maps
        img = img[0][0]
        if remove_pad:
            img = img[:, :, :Model.ori_shape[0], :Model.ori_shape[1]]  # [num_cams x height x width x channels]

        img_norm_cfg = Model.cfg.get('img_norm_cfg')
        mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
        mask = torch.Tensor(-mean)

        img_pert_list = []  # list of perturbed images
        for cam in range(len(attn_list)):
            img_og = img[cam].permute(1, 2, 0).numpy()

            # Get the attention for the camera and negate it if doing negative perturbation
            attn = attn_list[cam]
            if negative_pert:
                attn = -attn

            # Extract topk attention
            _, indices = torch.topk(attn.flatten(), k=num_tokens)
            indices = np.array(np.unravel_index(indices.numpy(), attn.shape)).T
            cols, rows = indices[:, 0], indices[:, 1]
            # Copy the normalized original images without bboxes
            img_pert = img_og.copy()

            # # Image perturbation by setting pixels to t (0,0,0)
            img_pert[cols, rows] = mask

            # Append the perturbed images to the list
            img_pert_list.append(img_pert)
        
        # save_img the perturbed 6 camera images into the data input
        img_pert_list = torch.from_numpy(np.stack(img_pert_list))
        img = [img_pert_list.permute(0, 3, 1, 2).unsqueeze(0)] # img[0] = torch.Size([1, 6, 3, 928, 1600])
        #img_compare = data['img'][0].data[0][0]
        data['img'][0] = DC(img)

        # Apply perturbated images to the model
        with torch.no_grad():
            output_pert = Model.model(return_loss=False, rescale=True, **data)

        outputs_pert.extend(output_pert)

        #del output_og
        del output_pert
        del attn_list
        gc.collect()
        torch.cuda.empty_cache()
        prog_bar.update()
    
    kwargs = {}
    eval_kwargs = Model.cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="bbox", **kwargs))
    eval_results = dataset.evaluate(outputs_pert, **eval_kwargs)
    mAP = eval_results['pts_bbox_NuScenes/mAP']
    NDS = eval_results['pts_bbox_NuScenes/NDS']

    with open(eval_file, "a") as file:
        file.write(f"Number of tokens: {num_tokens}\n")
        file.write(f"mAP: {mAP}\n")
        file.write(f"NDS: {NDS}\n")
        file.write("--------------------------\n")
    
    print(f"File {eval_file} updated.\n")
    gc.collect()
    del outputs_pert
    

if __name__ == '__main__':
    main()
