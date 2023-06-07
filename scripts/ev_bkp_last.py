from modules.Model import Model
from modules.Explainability import ExplainableTransformer
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
from PIL import Image
import torch
import numpy as np
import mmcv
import warnings
import cv2


def main():
    warnings.filterwarnings("ignore")

    expl_types = ["Attention Rollout", "Grad-CAM", "Gradient Rollout"]
    ObjectDetector = Model()
    ObjectDetector.load_from_config()
    ExplainabiliyGenerator = ExplainableTransformer(ObjectDetector)

    evaluate_expl(ObjectDetector, ExplainabiliyGenerator, expl_types[0], save=False)


def evaluate_expl(Model, ExplGen, expl_type, save=False):
    print(f"Evaluating {expl_type}...")

    bbox_idx = [0, 1, 2, 3, 4, 5, 6]
    pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    pred_threshold = 0.5
    layer = Model.layers - 1
    head_fusion, discard_ratio, raw_attention, handle_residual, apply_rule = \
        "min", 0.5, True, True, True

    outputs_pert = []

    dataset = Model.dataloader.dataset
    evaluation_lenght = len(dataset)
    initial_idx = 0
    num_tokens = int(pert_steps[1] * 1450)
    prog_bar = mmcv.ProgressBar(evaluation_lenght)

    for dataidx in range(initial_idx, initial_idx + evaluation_lenght):
        data = dataset[dataidx]
        metas = [[data['img_metas'][0].data]]
        img = [data['img'][0].data.unsqueeze(0)]  # img[0] = torch.Size([1, 6, 3, 928, 1600])
        data['img_metas'][0] = DC(metas, cpu_only=True)
        data['img'][0] = DC(img)

        # Attention scores are extracted, together with gradients if grad-CAM is selected
        if expl_type not in ["Grad-CAM", "Gradient Rollout"]:
            output_og = ExplGen.extract_attentions(data)
        else:
            output_og = ExplGen.extract_attentions(data, bbox_idx)

        nms_idxs = Model.model.module.pts_bbox_head.bbox_coder.get_indexes()

        # Extract predicted bboxes and their labels
        outputs = output_og[0]["pts_bbox"]
        img_metas = data["img_metas"][0]._data[0][0]
        thr_idxs = outputs['scores_3d'] > pred_threshold
        pred_bboxes = outputs["boxes_3d"][thr_idxs]
        pred_bboxes.tensor.detach()
        labels = outputs['labels_3d'][thr_idxs]

        img_norm_cfg = Model.cfg.get('img_norm_cfg')
        mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
        std = np.array(img_norm_cfg["std"], dtype=np.float32)

        bbox_idx = list(range(len(labels)))
        mask = torch.Tensor([0,0,0])
        attn_list = ExplGen.generate_explainability_cameras(expl_type, layer, bbox_idx, nms_idxs, head_fusion, discard_ratio, raw_attention, handle_residual, apply_rule, remove_pad=False)


        # Perturbate the input image with the XAI maps
        img = img[0][0]
        img_og_bboxes = []
        img_pert_list = []  # list of perturbed images
        for cam in range(len(attn_list)):
            attn = attn_list[cam]
            _, indices = torch.topk(attn.flatten(), k=num_tokens)
            indices = np.array(np.unravel_index(indices.numpy(), attn.shape)).T

            img_og = img[cam].permute(1, 2, 0).numpy()

            # Denormalize the images
            #img_og = mmcv.imdenormalize(img_og, mean, std, to_bgr=False)

            # Draw the og bboxes to the image for visualization
            # img_og_bb, _ = draw_lidar_bbox3d_on_img(
            #         pred_bboxes,
            #         img_og,
            #         img_metas['lidar2img'][cam],
            #         color=(0, 255, 0),
            #         with_label=True,
            #         mode_2d=True)
            # img_og_bboxes.append(img_og_bb)

            # Copy the normalized original images without bboxes
            img_pert = img_og.copy()
            # Image perturbation by setting pixels to (0,0,0)
            for idx in indices:
                img_pert[idx[0], idx[1]] = mask

            # Normalize the perturbed images and append to the list
            #img_pert = mmcv.imnormalize(img_pert, mean, std)
            img_pert_list.append(img_pert)

        # Save the perturbed 6 camera images into the data input
        img_pert_list = torch.Tensor(img_pert_list)
        img = [img_pert_list.permute(0, 3, 1, 2).unsqueeze(0)] # img[0] = torch.Size([1, 6, 3, 928, 1600])
        data['img'][0] = DC(img)

        # Apply perturbated images to the model
        with torch.no_grad():
            output_pert = Model.model(return_loss=False, rescale=True, **data)

        # outputs = output_pert[0]["pts_bbox"]
        # img_metas = data["img_metas"][0]._data[0][0]
        # thr_idxs = outputs['scores_3d'] > pred_threshold
        # pred_bboxes = outputs["boxes_3d"][thr_idxs]
        # pred_bboxes.tensor.detach()
        # labels = outputs['labels_3d'][thr_idxs]

        if save:
            # Create image with all 6 cameras
            hori = np.concatenate((img_og_bboxes[2], img_og_bboxes[0], img_og_bboxes[1]), axis=1)
            ver = np.concatenate((img_og_bboxes[5], img_og_bboxes[3], img_og_bboxes[4]), axis=1)
            img_og = np.concatenate((hori, ver), axis=0)

            hori = np.concatenate((img_pert_list[2], img_pert_list[0], img_pert_list[1]), axis=1)
            ver = np.concatenate((img_pert_list[5], img_pert_list[3], img_pert_list[4]), axis=1)
            img_pert = np.concatenate((hori, ver), axis=0)


            # Convert to RGB
            img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)
            img_pert = cv2.cvtColor(img_pert, cv2.COLOR_BGR2RGB)

            img_og = Image.fromarray(img_og)
            img_pert = Image.fromarray(img_pert)

            path_og = f"screenshots_eval/{Model.model_name}_{expl_type}_{dataidx}_og.png"
            path_pert = f"screenshots_eval/{Model.model_name}_{expl_type}_{dataidx}_pert.png"

            img_og.save(path_og)
            img_pert.save(path_pert)


        outputs_pert.extend(output_pert)
        torch.cuda.empty_cache()

        prog_bar.update()
    
    print("\nCompleted.\n")

    kwargs = {}
    eval_kwargs = Model.cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="bbox", **kwargs))
    print(dataset.evaluate(outputs_pert, **eval_kwargs))


def compare_arrays(array1, array2):
    comparison = np.not_equal(array1, array2)
    diff_indices = np.where(comparison)[0]
    count_diff = np.sum(comparison)
    return count_diff, diff_indices


if __name__ == '__main__':
    main()
