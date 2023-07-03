import torch
import cv2
import numpy as np


def handle_residual(orig_self_attention):
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

def avg_heads(attn, head_fusion="max"):
    if head_fusion == "mean":
        attn = attn.mean(dim=0)
    elif head_fusion == "max":
        attn = attn.max(dim=0)[0]
    elif head_fusion == "min":
        attn = attn.min(dim=0)[0]
    else:
        attn = attn[int(head_fusion)]

    return attn

def avg_heads_og(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

def gradcam(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    grad = grad.mean(dim=0, keepdim=True)
    cam = (cam * grad).mean(0).clamp(min=0)
    return cam


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
            
    def handle_co_attn_self_query(self, layer):
        # grad = self.dec_self_attn_grads[layer]
        cam = self.dec_self_attn_weights[layer]
        # cam = avg_heads_og(cam, grad)
        self.R_q_i += torch.matmul(cam, self.R_q_i)
        self.R_q_q += torch.matmul(cam, self.R_q_q)

    def handle_co_attn_query(self, layer, camidx, handle_residual_bool, apply_rule):
        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = avg_heads_og(cam_q_i, grad_q_i)
        R_qq_normalized = self.R_q_q
        if handle_residual_bool:
            R_qq_normalized = handle_residual(self.R_q_q)
        R_qi_addition = torch.matmul(R_qq_normalized.t(), cam_q_i)
        if not apply_rule:
            R_qi_addition = cam_q_i
        R_qi_addition[torch.isnan(R_qi_addition)] = 0
        self.R_q_i += R_qi_addition

    def extract_attentions(self, data, target_index=None):
        self.dec_cross_attn_weights, self.dec_cross_attn_grads, self.dec_self_attn_weights, self.dec_self_attn_grads = [], [], [], [] 

        hooks = []
        for layer in self.Model.model.module.pts_bbox_head.transformer.decoder.layers:
            hooks.append(
                layer.attentions[0].attn.register_forward_hook(
                    lambda _, input, output: self.dec_self_attn_weights.append(output[1][0].cpu())
                ))
            hooks.append(
                layer.attentions[1].attn.register_forward_hook(
                    lambda _, input, output: self.dec_cross_attn_weights.append(output[1].cpu())
                ))
        
        if self.height_feats is None:
            conv_feats = []
            hooks.append(
            self.Model.model.module.img_backbone.register_forward_hook(
                lambda _, input, output: conv_feats.append(output)
            ))

        if target_index is None:
            with torch.no_grad():
                outputs = self.Model.model(return_loss=False, rescale=True, all_layers=True, **data)

        else:
            for layer in self.Model.model.module.pts_bbox_head.transformer.decoder.layers:
                hooks.append(
                    layer.attentions[0].attn.register_backward_hook(
                        lambda _, grad_input, grad_output: self.dec_self_attn_grads.append(grad_input[0].permute(1, 0, 2)[0].cpu())
                    ))
     
            outputs = self.Model.model(return_loss=False, rescale=True, all_layers=True, **data)

            output_scores = outputs[0]["pts_bbox"][-1]["scores_3d"]
            one_hot = torch.zeros_like(output_scores).to(output_scores.device)
            one_hot[target_index] = 1
            one_hot.requires_grad_(True)
            one_hot = torch.sum(one_hot * output_scores)

            self.Model.model.zero_grad()
            one_hot.backward()
            for layer in self.Model.model.module.pts_bbox_head.transformer.decoder.layers:
                self.dec_cross_attn_grads.append(layer.attentions[1].attn.get_attn_gradients().cpu())

        if self.height_feats is None:
            if isinstance(conv_feats[0], dict):
                self.height_feats, self.width_feats = conv_feats[0]["stage5"].shape[-2:]   
            else:
                self.height_feats, self.width_feats = conv_feats[0][0].shape[-2:]
                                                                     
        for hook in hooks:
            hook.remove()
            
        return outputs
    
    def get_camera_scores(self):
        scores = []
        scores_perc = [] 

        for camidx in range(len(self.xai_maps[-1])):
            cam_map = self.xai_maps[-1][camidx]
            score = round(cam_map.sum().item(), 2)
            scores.append(score)

        sum_scores = sum(scores)
        if sum_scores > 0 and not np.isnan(sum_scores):
            for camidx in range(len(scores)):
                score_perc = round(((scores[camidx]/sum_scores)*100))
                scores_perc.append(score_perc)

        # for layer in range(len(self.xai_maps)):
        #     scores_cam = []
        #     for camidx in range(len(self.xai_maps[layer])):
        #         cam_map = self.xai_maps[layer][camidx]
        #         score = round(cam_map.sum().item(), 2)
        #         scores_cam.append(score)
        #     scores.append(scores_cam)
        
        # for layer in range(len(self.xai_maps)):
        #     sum_scores = sum(scores[layer])
        #     if sum_scores > 0 and not np.isnan(sum_scores):
        #         scores_perc_cam = []
        #         for camidx in range(len(scores[layer])):
        #             score_perc = round(((scores[layer][camidx]/sum_scores)*100))
        #             scores_perc_cam.append(score_perc)
        #         scores_perc.append(scores_perc_cam)
        #     else:
        #         continue
        return scores_perc
        
    def generate_raw_att(self, layer, camidx, head_fusion="min"):  
        ''' Generates Raw Attention for XAI. '''      

        # initialize relevancy matrices
        queries_num = self.dec_self_attn_weights[0].shape[-1]
        device = self.dec_cross_attn_weights[0].device

        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)

        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        cam_q_i = avg_heads(cam_q_i, head_fusion)
        
        self.R_q_i = cam_q_i  

    def generate_gradcam(self, layer, camidx):
        ''' Generates Grad-CAM for XAI. '''      

        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = gradcam(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
    
    def generate_gradroll(self, camidx, rollout=True, handle_residual=True, apply_rule=True):
        # initialize relevancy matrices

        queries_num = self.dec_self_attn_weights[0].shape[-1]
        image_bboxes = self.dec_cross_attn_weights[0].shape[-1]
        
        device = self.dec_cross_attn_weights[0].device

        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(device)

        # decoder self attention of queries followd by multi-modal attention
        if rollout:
            for layer in range(self.num_layers):
                self.handle_co_attn_self_query(layer)
                self.handle_co_attn_query(layer, camidx, handle_residual, apply_rule)
        else:
            self.handle_co_attn_self_query(self.num_layers-1)
            self.handle_co_attn_query(self.num_layers-1, camidx, handle_residual, apply_rule)

    def generate_explainability(self, expl_type, head_fusion="max", handle_residual=True, apply_rule=True):
        xai_maps, self_xai_maps, xai_maps_camera = [], [], []

        if expl_type == "Gradient Rollout":
            for camidx in range(6):
                self.generate_gradroll(camidx, handle_residual, apply_rule)
                xai_maps_camera.append(self.R_q_i.detach().cpu())
            xai_maps.append(xai_maps_camera)
            self_xai_maps = self.R_q_q.detach().cpu()

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

            self_attn_rollout = compute_rollout_attention(self.dec_self_attn_weights)
            self_xai_maps = self_attn_rollout.detach().cpu()

        # num_layers x num_cams x num_objects x 1450
        xai_maps = torch.stack([torch.stack(layer) for layer in xai_maps])
        xai_maps = xai_maps.permute(0, 2, 1, 3)  # num layers x num_objects x num_cams x 1450 # take only the selected objects

        # normalize across cameras
        for layer in range(len(xai_maps)):
            for object in range(len(xai_maps[layer])):
                xai_maps[layer][object] = (xai_maps[layer][object] - xai_maps[layer][object].min()) / (xai_maps[layer][object].max() - xai_maps[layer][object].min())

        self.xai_maps_full = xai_maps
        self.self_xai_maps_full = self_xai_maps

    def select_explainability(self, nms_idxs, bbox_idx, discard_threshold, maps_quality="Medium", remove_pad=True):
        self.xai_maps = self.xai_maps_full[:, nms_idxs[bbox_idx], :, :]
        self.self_xai_maps = self.self_xai_maps_full[nms_idxs[bbox_idx]][:, nms_idxs]

        # now attention maps can be overlayed
        if self.xai_maps.shape[1] > 0:
            self.xai_maps = self.xai_maps.max(dim=1)[0]  # num_layers x num_cams x [1450]
            mask = self.xai_maps < discard_threshold - (discard_threshold * 10) * (self.xai_maps.mean() + self.xai_maps.std())
            self.xai_maps[mask] = 0 
            self.xai_maps = self.interpolate_expl(self.xai_maps, maps_quality, remove_pad)

        if len(bbox_idx) == 1:
            self.scores = self.get_camera_scores()

    def interpolate_expl(self, xai_maps, maps_quality,remove_pad):
        xai_maps_inter = []
        if xai_maps.dim() == 1:
            xai_maps.unsqueeze_(0)
            xai_maps.unsqueeze_(0)
        interpol_res = {"Low": 8, "Medium": 16, "High": 32}
        for layer in range(len(xai_maps)):
            xai_maps_cameras = []
            for camidx in range(len(xai_maps[layer])):
                xai_map = xai_maps[layer][camidx].reshape(1, 1, self.height_feats, self.width_feats)
                xai_map[0,0,:,0] = 0
                xai_map[0,0,:,-1] = 0
                xai_map = torch.nn.functional.interpolate(xai_map, scale_factor=interpol_res[maps_quality], mode='bilinear')
                xai_map = xai_map.reshape(xai_map.shape[2], xai_map.shape[3])
                if remove_pad:
                    xai_map = xai_map[:self.ori_shape[0], :self.ori_shape[1]]

                xai_maps_cameras.append(xai_map)
            xai_maps_inter.append(xai_maps_cameras)

        xai_maps_inter = torch.stack([torch.stack(layer) for layer in xai_maps_inter])

        return xai_maps_inter


