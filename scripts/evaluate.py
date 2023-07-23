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
    expl_types = ['Raw Attention', 'Grad-CAM', 'Gradient Rollout', 'Random']

    ObjectDetector = Model()
    ObjectDetector.load_from_config()
    ExplainabiliyGenerator = ExplainableTransformer(ObjectDetector)

    evaluate(ObjectDetector, ExplainabiliyGenerator, expl_types[0], negative_pert=False, pred_threshold=0.4, remove_pad=True)


def evaluate(Model, ExplGen, expl_type, negative_pert=False, pred_threshold=0.1, remove_pad=True):
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

    pert_steps = [0]

    print(info)
    start_time = time.time()
    for step in range(len(pert_steps)):
        perc = pert_steps[step] * 100
        print(f"\nNumber of tokens removed: {perc} %")
        evaluate_step(Model, ExplGen, expl_type, step=pert_steps[step], negative_pert=negative_pert, eval_file=file_path, remove_pad=remove_pad, pred_threshold=pred_threshold)
        gc.collect()
        torch.cuda.empty_cache()
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Completed (elapsed time {total_time} seconds).\n")

    with open(file_path, "a") as file:
        file.write("--------------------------\n")
        file.write(f"Elapsed time: {total_time}\n")

def evaluate_step(Model, ExplGen, expl_type, step, eval_file, negative_pert=True, pred_threshold=0.1, remove_pad=False):

    head_fusion, discard_threshold, handle_residual, apply_rule = \
        "max", 0.3, True, True
    
    dataset = Model.dataset
    evaluation_lenght = len(dataset)
    outputs_pert = []

    Model.model.eval()

    prog_bar = mmcv.ProgressBar(evaluation_lenght)

    for dataidx in range(evaluation_lenght):
        data = dataset[dataidx]
        metas = [[data['img_metas'][0].data]]
        img = [data['img'][0].data.unsqueeze(0)]  # img[0] = torch.Size([1, 6, 3, 928, 1600])
        data['img_metas'][0] = DC(metas, cpu_only=True)
        data['img'][0] = DC(img)
    
        if "points" in data.keys():
            data.pop("points")

        # if expl_type != "Random":
        #     # Attention scores are extracted, together with gradients if grad-CAM is selected
        #     output_og = ExplGen.extract_attentions(data)

        #     # Extract predicted bboxes and their labels
        #     outputs = output_og[0]["pts_bbox"]
        #     nms_idxs = Model.model.module.pts_bbox_head.bbox_coder.get_indexes().cpu()
        #     thr_idxs = outputs['scores_3d'] > pred_threshold
        #     labels = outputs['labels_3d'][thr_idxs]
            
        #     bbox_idx = list(range(len(labels)))
        #     if expl_type in ["Grad-CAM", "Gradient Rollout"]:
        #         ExplGen.extract_attentions(data, bbox_idx)
        #     ExplGen.generate_explainability(expl_type, head_fusion, handle_residual, apply_rule)
        #     ExplGen.select_explainability(nms_idxs, bbox_idx, discard_threshold, maps_quality="High", remove_pad=True)
        #     xai_maps = ExplGen.xai_maps.max(dim=0)[0]
        
        # else:
        #     if remove_pad:
        #         xai_maps = torch.rand(6, Model.ori_shape[0], Model.ori_shape[1])
        #     else:
        #         xai_maps = torch.rand(6, Model.pad_shape[0], Model.pad_shape[1])

        # # Perturbate the input image with the XAI maps
        # img = img[0][0]
        # if remove_pad:
        #     img = img[:, :, :Model.ori_shape[0], :Model.ori_shape[1]]  # [num_cams x height x width x channels]

        # img_norm_cfg = Model.cfg.get('img_norm_cfg')
        # mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
        # mask = torch.Tensor(-mean)

        # # defect data: 475
        # img_pert_list = []  # list of perturbed images
        # for cam in range(len(xai_maps)):
        #     img_pert = img[cam].permute(1, 2, 0).numpy()

        #     # Get the attention for the camera and negate it if doing negative perturbation
        #     xai_cam = xai_maps[cam]
        #     filter_mask = xai_cam > 0.2
        #     filtered_xai = xai_cam[filter_mask].flatten()
        #     original_indices = torch.arange(xai_cam.numel()).reshape(xai_cam.shape)[filter_mask].flatten()
        #     if negative_pert:
        #         xai_cam = -xai_cam
                # filtered_xai = - filtered_xai

        #     top_k = int(step * filtered_xai.numel())
        #     _, indices = torch.topk(filtered_xai, top_k)
        #     original_indices = original_indices[indices]
        #     row_indices, col_indices = original_indices // xai_cam.size(1), original_indices % xai_cam.size(1)
        #     img_pert[row_indices, col_indices] = mask

        #     img_pert_list.append(img_pert)
        
        # # if there are detected objects, apply the perturbated images to the data input
        # if len(img_pert_list) > 0:
        #     # save_img the perturbed 6 camera images into the data input
        #     img_pert_list = torch.from_numpy(np.stack(img_pert_list))
        #     img = [img_pert_list.permute(0, 3, 1, 2).unsqueeze(0)] # img = [torch.Size([1, 6, 3, 928, 1600])
        #     data['img'][0] = DC(img)

        # Apply perturbated images to the model
        with torch.no_grad():
            output_pert = Model.model(return_loss=False, rescale=True, **data)

        outputs_pert.extend(output_pert)

        #del output_og
        del output_pert
        # del xai_maps
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
