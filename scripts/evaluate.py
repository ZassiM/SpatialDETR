from modules.Model import Model
from modules.Explainability import ExplainableTransformer
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np
import mmcv
import warnings


def main():
    warnings.filterwarnings("ignore")

    expl_types = ["Attention Rollout", "Grad-CAM", "Gradient Rollout"]

    ObjectDetector = Model()
    ObjectDetector.load_from_config()
    ExplainabiliyGenerator = ExplainableTransformer(ObjectDetector)
    evaluate_expl(ObjectDetector, ExplainabiliyGenerator, expl_types[0])


def evaluate_expl(Model, ExplGen, expl_type):
    print(f"Evaluating {expl_type}...")

    bbox_idx = [0]
    layer = Model.layers - 1
    head_fusion, discard_ratio, raw_attention, handle_residual, apply_rule = \
        "min", 0.5, True, True, True

    outputs_pert = []

    dataset = Model.dataloader.dataset
    evaluation_lenght = len(dataset)
    initial_idx = 0
    num_tokens = int(0.75 * 1450)
    prog_bar = mmcv.ProgressBar(evaluation_lenght)

    for i in range(initial_idx, initial_idx + evaluation_lenght):
        data = dataset[i]
        metas = [[data['img_metas'][0].data]]
        img = [data['img'][0].data.unsqueeze(0)]  # img[0] = torch.Size([1, 6, 3, 928, 1600])
        data['img_metas'][0] = DC(metas, cpu_only=True)
        data['img'][0] = DC(img)

        # Attention scores are extracted, together with gradients if grad-CAM is selected
        if expl_type not in ["Grad-CAM", "Gradient Rollout"]:
            ExplGen.extract_attentions(data)
        else:
            ExplGen.extract_attentions(data, bbox_idx)

        nms_idxs = Model.model.module.pts_bbox_head.bbox_coder.get_indexes()

        topk_list = []
        
        attn_list = ExplGen.generate_explainability_cameras(expl_type, layer, bbox_idx, nms_idxs, head_fusion, discard_ratio, raw_attention, handle_residual, apply_rule)

        for i in range(len(attn_list)):
            attn = attn_list[i]
            _, indices = torch.topk(attn.flatten(), k=num_tokens)
            indices = np.array(np.unravel_index(indices.numpy(), attn.shape)).T
            topk_list.append(indices)

        img_og_list = []  # list of original images
        img_pert_list = []  # list of perturbed images

        img_norm_cfg = Model.cfg.get('img_norm_cfg')
        mean = np.array(img_norm_cfg["mean"], dtype=np.float32)
        std = np.array(img_norm_cfg["std"], dtype=np.float32)
        mask = (-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2])

        # Denormalization is needed, because data is normalized
        img = img[0][0]
        img = img[:, :, :Model.ori_shape[0], :Model.ori_shape[1]]
        for i in range(len(img)):
            img_og = img[i].permute(1, 2, 0)
            img_og_list.append(img_og)
            img_pert = img_og.clone()
            # Image perturbation by setting pixels to (0,0,0)
            for idx in topk_list[i]:
                img_pert[idx[0], idx[1]] = mask
            img_pert_list.append(img_pert.permute(2, 0, 1))

        # Save the perturbed 6 camera images into the data input
        img = [torch.stack((img_pert_list)).unsqueeze(0)]  # img[0] = torch.Size([1, 6, 3, 928, 1600])
        data['img'][0] = DC(img)

        # Second forward
        with torch.no_grad():
            output = Model.model(return_loss=False, rescale=True, **data)

        outputs_pert.extend(output)
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


if __name__ == '__main__':
    main()
