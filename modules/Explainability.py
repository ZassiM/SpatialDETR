import torch
import numpy as np


# normalization- eq. 8+9
def normalize_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    matrices_aug = all_layer_matrices
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention



class ExplainableTransformer:
    '''
    Generates attention maps and different explainability features.
    '''
    def __init__(self, Model):
        self.Model = Model
        self.num_layers = Model.num_layers
        self.ori_shape = Model.ori_shape
        self.Model.model.eval()
        self.height_feats, self.width_feats = None, None


    def extract_attentions(self, data, target_index=None):
        hooks = []

        if self.height_feats is None:
            conv_feats = []
            hooks.append(
            self.Model.model.module.img_backbone.register_forward_hook(
                lambda _, input, output: conv_feats.append(output)
            ))

        if target_index is None:
            self.dec_cross_attn_weights, self.dec_self_attn_weights = [], []
            for layer in self.Model.model.module.pts_bbox_head.transformer.decoder.layers:
                hooks.append(
                    layer.attentions[0].attn.register_forward_hook(
                        lambda _, input, output: self.dec_self_attn_weights.append(output[1].cpu())
                    ))
                hooks.append(
                    layer.attentions[1].attn.register_forward_hook(
                        lambda _, input, output: self.dec_cross_attn_weights.append(output[1].cpu())
                    ))
            with torch.no_grad():
                outputs = self.Model.model(return_loss=False, rescale=True, **data)

        else:
            self.dec_cross_attn_grads, self.dec_self_attn_grads = [], []

            outputs = self.Model.model(return_loss=False, rescale=True, **data)

            output_scores = outputs[0]["pts_bbox"]["scores_3d"]
            one_hot = torch.zeros_like(output_scores).to(output_scores.device)
            one_hot[target_index] = 1
            one_hot.requires_grad_(True)
            one_hot = torch.sum(one_hot * output_scores)

            self.Model.model.zero_grad()
            one_hot.backward()

            # retrieve gradients
            for layer in self.Model.model.module.pts_bbox_head.transformer.decoder.layers:
                self.dec_cross_attn_grads.append(layer.attentions[1].attn.get_attn_gradients().cpu())

            for layer in self.Model.model.module.pts_bbox_head.transformer.decoder.layers:
                self.dec_self_attn_grads.append(layer.attentions[0].attn.get_attn_gradients().cpu())

        if self.height_feats is None:
            if isinstance(conv_feats[0], dict):
                self.height_feats, self.width_feats = conv_feats[0]["stage5"].shape[-2:]
            else:
                self.height_feats, self.width_feats = conv_feats[0][0].shape[-2:]

        for hook in hooks:
            hook.remove()

        return outputs


    def generate_explainability(self, expl_type, head_fusion="max", handle_residual=True, apply_rule=True):
        xai_maps, self.self_xai_maps_full, xai_maps_camera = [], [], []

        if expl_type == "Gradient Rollout":
            for camidx in range(6):
                self.generate_gradroll(camidx, apply_normalization=handle_residual, apply_rule=apply_rule)
                xai_maps_camera.append(self.R_q_i.detach().cpu())
            xai_maps.append(xai_maps_camera)

        else:
            for layer in range(self.num_layers):
                xai_maps_camera = []
                for camidx in range(6):
                    if expl_type == "Raw Attention":
                        self.generate_raw_att(layer, camidx, head_fusion)
                    elif expl_type == "Grad-CAM":
                        self.generate_gradcam(layer, camidx)
                    xai_maps_camera.append(self.R_q_i.detach().cpu())
                xai_maps.append(xai_maps_camera)

        if expl_type in ["Self Attention", "Gradient Rollout"]:
            for layer in range(self.num_layers):
                # 1 x num_heads x num_queries x num_queries
                self.self_xai_maps_full.append(self.dec_self_attn_weights[layer].squeeze().clamp(min=0).mean(dim=0))

        # num_layers x num_cams x num_queries x 1450
        xai_maps = torch.stack([torch.stack(layer) for layer in xai_maps])
        # num layers x num_queries x num_cams x 1450
        xai_maps = xai_maps.permute(0, 2, 1, 3)

        # normalize across cameras
        for layer in range(len(xai_maps)):
            for query in range(len(xai_maps[layer])):
                xai_maps[layer][query] = (xai_maps[layer][query] - xai_maps[layer][query].min()) / (xai_maps[layer][query].max() - xai_maps[layer][query].min())

        self.xai_maps_full = xai_maps

    def select_explainability(self, nms_idxs, bbox_idx, discard_threshold, maps_quality="Medium", remove_pad=True, layer_fusion_method="max"):
        self.self_xai_maps = []
        for layer in range(len(self.self_xai_maps_full)):
            self.self_xai_maps.append(self.self_xai_maps_full[layer][nms_idxs[bbox_idx]][:, nms_idxs][0])

        # apply discrod threshold and interpolate xai_maps
        self.xai_maps = self.xai_maps_full[:, nms_idxs[bbox_idx], :, :]
        if self.xai_maps.shape[1] > 0:
            self.xai_maps = self.xai_maps.max(dim=1)[0]  # num_layers x num_cams x [1450]
            mask = self.xai_maps < discard_threshold - (discard_threshold * 10) * (self.xai_maps.mean() + self.xai_maps.std())
            self.xai_maps[mask] = 0
            self.xai_maps = self.interpolate(self.xai_maps, maps_quality, remove_pad)

        # store original per layer maps
        self.xai_layer_maps = self.xai_maps

        # fuse layers with fusion algorithms
        if layer_fusion_method == "max":
            self.xai_maps = self.xai_maps.max(dim=0, keepdim=True)[0]
        elif layer_fusion_method == "zero_clamp_mean":
            self.xai_maps = self.xai_maps.clamp(min=0).mean(dim=0, keepdim=True)
        elif layer_fusion_method == "mean":
            self.xai_maps = self.xai_maps.mean(dim=0, keepdim=True)
        elif layer_fusion_method == "min":
            self.xai_maps = self.xai_maps.min(dim=0, keepdim=True)[0]
        elif layer_fusion_method == "last":
            self.xai_maps = self.xai_maps[-1, ...]
        else:
            raise NotImplementedError

        self.xai_maps = self.xai_maps.squeeze()

        # if only one box selected
        if len(bbox_idx) == 1:
            self.scores = self.get_camera_scores()


    def interpolate(self, xai_maps, maps_quality, remove_pad):
        xai_maps_inter = []
        if xai_maps.dim() == 1:
            xai_maps.unsqueeze_(0)
            xai_maps.unsqueeze_(0)
        interpol_res = {"Low": 8, "Medium": 16, "High": 32}
        for layer in range(len(xai_maps)):
            xai_maps_cameras = []
            for camidx in range(len(xai_maps[layer])):
                xai_map = xai_maps[layer][camidx].reshape(1, 1, self.height_feats, self.width_feats)
                # xai_map[0,0,:,0] = 0
                # xai_map[0,0,:,-1] = 0# Assume 'tensor' is your input tensor
                xai_map = xai_map.roll(-1, dims=-1)
                xai_map[0, 0, :, -1] = 0
                xai_map = torch.nn.functional.interpolate(xai_map, scale_factor=interpol_res[maps_quality], mode='bilinear')
                xai_map = xai_map.reshape(xai_map.shape[2], xai_map.shape[3])
                if remove_pad:
                    xai_map = xai_map[:self.ori_shape[0], :self.ori_shape[1]]

                xai_maps_cameras.append(xai_map)
            xai_maps_inter.append(xai_maps_cameras)

        xai_maps_inter = torch.stack([torch.stack(layer) for layer in xai_maps_inter])

        return xai_maps_inter


    def get_camera_scores(self):
        scores = []
        scores_perc = []

        for camidx in range(len(self.xai_layer_maps[-1])):
            cam_map = self.xai_layer_maps[-1][camidx]
            score = round(cam_map.sum().item(), 2)
            scores.append(score)

        sum_scores = sum(scores)
        if sum_scores > 0 and not np.isnan(sum_scores):
            for camidx in range(len(scores)):
                score_perc = round(((scores[camidx]/sum_scores)*100))
                scores_perc.append(score_perc)

        return scores_perc

    def generate_raw_att(self, layer, camidx, head_fusion_method="max"):
        cam_q_i = self.dec_cross_attn_weights[layer][camidx]

        if head_fusion_method == "mean":
            cam_q_i = cam_q_i.mean(dim=0)
        elif head_fusion_method == "zero_clamp_mean":
            cam_q_i = cam_q_i.clamp(min=0).mean(dim=0)
        elif head_fusion_method == "max":
            cam_q_i = cam_q_i.max(dim=0)[0]
        elif head_fusion_method == "min":
            cam_q_i = cam_q_i.min(dim=0)[0]
        elif head_fusion_method.isdigit():
            cam_q_i = cam_q_i[int(head_fusion_method)]
        else:
            raise NotImplementedError

        self.R_q_i = cam_q_i

    def generate_gradcam(self, layer, camidx):
        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = (cam_q_i * grad_q_i).mean(dim=0).clamp(min=0)
        self.R_q_i = cam_q_i

    # rule 5 from paper
    def zero_clamp_avg_grad_heads(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        return cam

    def handle_co_attn_self_query(self, layer):
        cam_qq = self.dec_self_attn_weights[layer]
        grad = self.dec_self_attn_grads[layer]
        cam_qq = self.zero_clamp_avg_grad_heads(cam_qq, grad)
        self.R_q_q += torch.matmul(cam_qq, self.R_q_q)
        self.R_q_i += torch.matmul(cam_qq, self.R_q_i)

    def handle_co_attn_query(self, layer, camidx, apply_normalization, apply_rule):
        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = self.zero_clamp_avg_grad_heads(cam_q_i, grad_q_i)

        if apply_normalization:
            R_q_i_addition = torch.matmul(normalize_residual(self.R_q_q).t(), cam_q_i)
        else:
            R_q_i_addition = torch.matmul(self.R_q_q.t(), cam_q_i)

        if not apply_rule:
            R_qi_addition = cam_q_i
        R_qi_addition[torch.isnan(R_qi_addition)] = 0
        self.R_q_i += R_q_i_addition

    def generate_gradroll(self, camidx, rollout=True, apply_normalization=True, apply_rule=True):
        # initialize relevancy matrices

        queries_num = self.dec_self_attn_weights[0].shape[-1]
        image_tokens = self.dec_cross_attn_weights[0].shape[-1]

        device = self.dec_cross_attn_weights[0].device

        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_tokens).to(device)

        # decoder self attention of queries followd by multi-modal attention
        if rollout:
            for layer in range(self.num_layers):
                self.handle_co_attn_self_query(layer)
                self.handle_co_attn_query(layer, camidx, apply_normalization, apply_rule)
        else:
            self.handle_co_attn_self_query(-1)
            self.handle_co_attn_query(-1, camidx, apply_normalization, apply_rule)

