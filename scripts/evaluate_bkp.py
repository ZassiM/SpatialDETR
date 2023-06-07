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

    evaluate_expl(ObjectDetector, ExplainabiliyGenerator, expl_types[0], save=True)


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
            outputs = ExplGen.extract_attentions(data)
        else:
            outputs = ExplGen.extract_attentions(data, bbox_idx)

        nms_idxs = Model.model.module.pts_bbox_head.bbox_coder.get_indexes()

        # Extract predicted bboxes and their labels
        outputs = outputs[0]["pts_bbox"]
        img_metas = data["img_metas"][0]._data[0][0]
        thr_idxs = outputs['scores_3d'] > pred_threshold
        pred_bboxes = outputs["boxes_3d"][thr_idxs]
        pred_bboxes.tensor.detach()
        labels = outputs['labels_3d'][thr_idxs]

        bbox_idx = list(range(len(labels)))
        attn_list = ExplGen.generate_explainability_cameras(expl_type, layer, bbox_idx, nms_idxs, head_fusion, discard_ratio, raw_attention, handle_residual, apply_rule, remove_pad=False)

        topk_list = []
        for cam in range(len(attn_list)):
            attn = attn_list[cam]
            _, indices = torch.topk(attn.flatten(), k=num_tokens)
            indices = np.array(np.unravel_index(indices.numpy(), attn.shape)).T
            topk_list.append(indices)

        img_og_list = []  # list of original images
        img_pert_list = []  # list of perturbed images

        img_norm_cfg = Model.cfg.get('img_norm_cfg')
        mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
        std = np.array(img_norm_cfg["std"], dtype=np.float32)
        
        img = img[0][0]
        for camidx in range(len(img)):
            img_og = img[camidx].permute(1, 2, 0).numpy()

            img_og, _ = draw_lidar_bbox3d_on_img(
                    pred_bboxes,
                    img_og,
                    img_metas['lidar2img'][camidx],
                    color=(0, 255, 0),
                    mode_2d=True)

            img_og_list.append(img_og)
            img_pert = img_og.copy()
            mask = torch.Tensor([img_og[:, :, 0].min().item(), img_og[:, :, 1].min().item(), img_og[:, :, 2].min().item()])
            for idx in topk_list[camidx]:
                img_pert[idx[0], idx[1]] = mask
            #img_pert_list.append(img_pert.permute(2, 0, 1))
            img_pert_list.append(img_pert)

        if save:
            # Create image with all 6 cameras
            hori = np.concatenate((img_og_list[2], img_og_list[0], img_og_list[1]), axis=1)
            ver = np.concatenate((img_og_list[5], img_og_list[3], img_og_list[4]), axis=1)
            img_og = np.concatenate((hori, ver), axis=0)

            hori = np.concatenate((img_pert_list[2], img_pert_list[0], img_pert_list[1]), axis=1)
            ver = np.concatenate((img_pert_list[5], img_pert_list[3], img_pert_list[4]), axis=1)
            img_pert = np.concatenate((hori, ver), axis=0)

            # Denormalize
            img_og = mmcv.imdenormalize(img_og, mean, std, to_bgr=False)
            img_pert = mmcv.imdenormalize(img_pert, mean, std, to_bgr=False)

            # Convert to RGB
            img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB)
            img_pert = cv2.cvtColor(img_pert, cv2.COLOR_BGR2RGB)

            img_og = Image.fromarray(img_og.astype(np.uint8))
            img_pert = Image.fromarray(img_pert.astype(np.uint8))

            path_og = f"screenshots_eval/{Model.model_name}_{expl_type}_{dataidx}_og.png"
            path_pert = f"screenshots_eval/{Model.model_name}_{expl_type}_{dataidx}_pert.png"

            img_og.save(path_og)
            img_pert.save(path_pert)


        # for cam in range(len(img)):
        #     img_og = img[cam].permute(1, 2, 0)
        #     img_og_list.append(img_og)
        #     img_pert = img_og.clone()
        #     mask = torch.Tensor([img_og[:, :, 0].min().item(), img_og[:, :, 1].min().item(), img_og[:, :, 2].min().item()])
        #     # Image perturbation by setting pixels to (0,0,0)
        #     for idx in topk_list[cam]:
        #         img_pert[idx[0], idx[1]] = mask
        #     img_pert_list.append(img_pert.permute(2, 0, 1))

        # # Save the perturbed 6 camera images into the data input
        # img = [torch.stack((img_pert_list)).unsqueeze(0)]  # img[0] = torch.Size([1, 6, 3, 928, 1600])
        # data['img'][0] = DC(img)

        # # Second forward
        # with torch.no_grad():
        #     output = Model.model(return_loss=False, rescale=True, **data)

        # outputs_pert.extend(output)
        # torch.cuda.empty_cache()

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


if __name__ == '__main__':
    main()
