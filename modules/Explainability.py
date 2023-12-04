import torch
import numpy as np


# Self attention maps normalization function (c.f. Hila Chefer paper)
def normalize_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention

class ExplainableTransformer:
    '''
    Generates attention maps and different explainability methods.
    '''
    def __init__(self, Model):
        self.Model = Model
        self.num_layers = Model.num_layers
        self.ori_shape = Model.ori_shape
        self.Model.model.eval()
        self.height_feats, self.width_feats = None, None

    def extract_attentions(self, data, target_index=None):
        ''' Extracts the attention maps with PyTorch hooks during forward propagation, or backpropagation if gradients are needed. '''
        hooks = []

        if self.height_feats is None:
            conv_feats = []
            hooks.append(
            self.Model.model.module.img_backbone.register_forward_hook(
                lambda _, input, output: conv_feats.append(output)
            ))

        if target_index is None:
            # Only forward pass is needed
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
            # Forwad + Backpropagation for gradients extraction
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
            # Extract the CNN height and width of the images
            if isinstance(conv_feats[0], dict):
                self.height_feats, self.width_feats = conv_feats[0]["stage5"].shape[-2:]
            else:
                self.height_feats, self.width_feats = conv_feats[0][0].shape[-2:]

        for hook in hooks:
            hook.remove()

        return outputs

    def generate_explainability(self, expl_type, head_fusion="max", handle_residual=True, apply_rule=True):
        ''' Generates a selected XAI technique from the extracted attention maps.
            This function runs only when one of its parameters are changed, by using the Config class'''
        xai_maps, self.self_xai_maps_full, xai_maps_camera = [], [], []

        # Gradient Rollout: uses by design all the Transformer layers
        if expl_type == "Gradient Rollout":
            for camidx in range(6):
                self.generate_gradroll(camidx, apply_normalization=handle_residual, apply_rule=apply_rule)
                xai_maps_camera.append(self.R_q_i.detach().cpu())
            xai_maps.append(xai_maps_camera)

        # Raw Attention and Grad-CAM: the generated xai_maps will contain all layers
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

        # Self-Attention and Gradient Rollout use the decoder self attention maps
        if expl_type in ["Self Attention", "Gradient Rollout"]:
            for layer in range(self.num_layers):
                # Shape: 1 x num_heads x num_queries x num_queries
                self.self_xai_maps_full.append(self.dec_self_attn_weights[layer].squeeze().clamp(min=0).mean(dim=0))

        # Shape: num_layers x num_cams x num_queries x 1450
        xai_maps = torch.stack([torch.stack(layer) for layer in xai_maps])
        # Shape: num layers x num_queries x num_cams x 1450
        xai_maps = xai_maps.permute(0, 2, 1, 3)

        # Normalize across cameras
        for layer in range(len(xai_maps)):
            for query in range(len(xai_maps[layer])):
                xai_maps[layer][query] = (xai_maps[layer][query] - xai_maps[layer][query].min()) / (xai_maps[layer][query].max() - xai_maps[layer][query].min())

        self.xai_maps_full = xai_maps

    def select_explainability(self, nms_idxs, bbox_idx, discard_threshold, maps_quality="Medium", remove_pad=True, layer_fusion_method="max"):
        ''' After the XAI maps have been generated, this function applies some user-defined parameters/filters to the xai_maps object.
            For instance, only the objects present in the image are considered (by using the nms_idxs parameter). '''
        
        # Final self-attention maps
        self.self_xai_maps = []
        for layer in range(len(self.self_xai_maps_full)):
            self.self_xai_maps.append(self.self_xai_maps_full[layer][nms_idxs[bbox_idx]][:, nms_idxs][0])

        # Apply discard threshold and interpolate xai_maps
        self.xai_maps = self.xai_maps_full[:, nms_idxs[bbox_idx], :, :]
        if self.xai_maps.shape[1] > 0:
            self.xai_maps = self.xai_maps.max(dim=1)[0]  # num_layers x num_cams x [1450]
            mask = self.xai_maps < discard_threshold - (discard_threshold * 10) * (self.xai_maps.mean() + self.xai_maps.std())
            self.xai_maps[mask] = 0
            self.xai_maps = self.interpolate(self.xai_maps, maps_quality, remove_pad)

        # Store original per layer maps before fusing them
        self.xai_layer_maps = self.xai_maps

        # Fuse layers with fusion algorithms
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

        # if only one box is selected, extract a  pseudo-confidence score of the model
        if len(bbox_idx) == 1:
            self.scores = self.get_camera_scores()

    def interpolate(self, xai_maps, maps_quality, remove_pad):
        ''' XAI maps interpolation function: based on user-defined maps quality and the boolean remove_pad,
            it interpolated the maps 8/16/32 times and maintains the image original shape if remove pad is true.'''
        
        xai_maps_inter = []
        if xai_maps.dim() == 1:
            xai_maps.unsqueeze_(0)
            xai_maps.unsqueeze_(0)
        interpol_res = {"Low": 8, "Medium": 16, "High": 32}
        for layer in range(len(xai_maps)):
            xai_maps_cameras = []
            for camidx in range(len(xai_maps[layer])):
                xai_map = xai_maps[layer][camidx].reshape(1, 1, self.height_feats, self.width_feats)
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
        ''' It extracts a score from the xai_maps depending on their values through each camera.
            The higher the sum of the values, the higher the score. '''
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
        ''' Generates the Raw Attention XAI maps. '''

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
        ''' Generates the Grad-CAM XAI maps (inspiration form Hila Chefer implementation). '''

        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = (cam_q_i * grad_q_i).mean(dim=0).clamp(min=0)
        self.R_q_i = cam_q_i

    def generate_gradroll(self, camidx, rollout=True, apply_normalization=True, apply_rule=True):
        ''' Generates the Gradient-Rollout XAI maps (inspiration form Hila Chefer implementation). '''

        # Initialize Relevancy matrices
        queries_num = self.dec_self_attn_weights[0].shape[-1]
        image_tokens = self.dec_cross_attn_weights[0].shape[-1]

        device = self.dec_cross_attn_weights[0].device

        # Queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)

        # Queries cross attention matrix
        self.R_q_i = torch.zeros(queries_num, image_tokens).to(device)

        # By default, this technique loops through all Transformer layers and generates an unified XAI map.
        if rollout:
            for layer in range(self.num_layers):
                self.handle_co_attn_self_query(layer)
                self.handle_co_attn_query(layer, camidx, apply_normalization, apply_rule)
        else:
            self.handle_co_attn_self_query(-1)
            self.handle_co_attn_query(-1, camidx, apply_normalization, apply_rule)

    def zero_clamp_avg_grad_heads(self, cam, grad):
        ''' Used for clamping the attention maps during Gradient Rollout calculation. '''
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        return cam

    def handle_co_attn_self_query(self, layer):
        ''' Used for updating the relevancy maps on Transformer self-attention layers.
            Only for Gradient Rollout. '''
        cam_qq = self.dec_self_attn_weights[layer]
        grad = self.dec_self_attn_grads[layer]
        cam_qq = self.zero_clamp_avg_grad_heads(cam_qq, grad)
        self.R_q_q += torch.matmul(cam_qq, self.R_q_q)
        self.R_q_i += torch.matmul(cam_qq, self.R_q_i)

    def handle_co_attn_query(self, layer, camidx, apply_normalization, apply_rule):
        ''' Used for updating the relevancy maps on Transformer cross-attention layers.
            Only for Gradient Rollout. '''
        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = self.zero_clamp_avg_grad_heads(cam_q_i, grad_q_i)

        if apply_normalization:
            R_q_i_addition = torch.matmul(normalize_residual(self.R_q_q).t(), cam_q_i)
        else:
            R_q_i_addition = torch.matmul(self.R_q_q.t(), cam_q_i)

        if not apply_rule:
            R_q_i_addition = cam_q_i
        R_q_i_addition[torch.isnan(R_q_i_addition)] = 0
        self.R_q_i += R_q_i_addition

