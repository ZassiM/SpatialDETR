import numpy as np
import torch

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
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

# rule 5 from paper
def avg_heads(attn, grad):
    attn = attn.reshape(-1, attn.shape[-2], attn.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    attn = grad * attn
    attn = attn.clamp(min=0).mean(dim=0)
    return attn

def avg_heads(attn, head_fusion = "min", discard_ratio = 0.9):
    if head_fusion == "mean":
        attn = attn.mean(dim=0)
    elif head_fusion == "max":
        attn = attn.max(dim=0)[0]
    elif head_fusion == "min":
        attn = attn.min(dim=0)[0]

    flat = attn.view(attn.size(0), -1)
    #1450*discard_ratio smallest elements 
    _, indices = flat.topk(int(attn.size(-1)*discard_ratio), -1, False)
    for i in range(len(indices)):
        flat[i, indices[i]] = 0
    return attn

# rules 6 + 7 from paper
def apply_self_attention_rules(R_ss, R_sq, attn_ss):
    R_sq_addition = torch.matmul(attn_ss, R_sq)
    R_ss_addition = torch.matmul(attn_ss, R_ss)
    return R_ss_addition, R_sq_addition

# rule 10 from paper
def apply_mm_attention_rules(R_ss, R_qq, attn_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        #R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized, torch.matmul(attn_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = attn_sq
    R_sq_addition[torch.isnan(R_sq_addition)] = 0
    return R_sq_addition

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention


class Generator:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.camidx = None
        self.layers = 0
        for _ in self.model.module.pts_bbox_head.transformer.decoder.layers:
            self.layers += 1
        self.layer = self.layers - 1
        self.dec_cross_attn_weights, self.dec_cross_attn_grads, self.dec_self_attn_weights, self.dec_self_attn_grads = [], [], [], []    
    
    def extract_attentions(self, data, target_index = None):
        
        self.dec_self_attn_weights, self.dec_cross_attn_weights = [], []
        hooks = []
        for layer in self.model.module.pts_bbox_head.transformer.decoder.layers:
            hooks.append(
            layer.attentions[0].attn.register_forward_hook(
                lambda _, input, output: self.dec_self_attn_weights.append(output[1][0])
            ))
            # SpatialDETR
            if hasattr(layer.attentions[1], "attn"):
                hooks.append(
                layer.attentions[1].attn.register_forward_hook(
                    lambda _, input, output: self.dec_cross_attn_weights.append(output[1])
                ))
            # DETR3D
            else:
                hooks.append(
                layer.attentions[1].attention_weights.register_forward_hook(
                    lambda _, input, output: self.dec_cross_attn_weights.append(output)
                ))

        if "points" in data.keys():
            data.pop("points")
            
        if target_index is None:
            with torch.no_grad():
                outputs = self.model(return_loss=False, rescale=True, **data)
        
        else:
            outputs = self.model(return_loss=False, rescale=True, **data)
            
            output_scores = outputs[0]["pts_bbox"]["scores_3d"]
            one_hot = torch.zeros_like(output_scores).to(output_scores.device)
            one_hot[target_index] = 1
            one_hot.requires_grad_(True)
            one_hot = torch.sum(one_hot * output_scores)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)
            
            for layer in self.model.module.pts_bbox_head.transformer.decoder.layers:
                self.dec_cross_attn_grads.append(layer.attentions[1].attn.get_attn_gradients())
        
        for hook in hooks:
            hook.remove()
            
        return outputs

    def get_all_attn(self, target_idx, indexes, head_fusion = "min", discard_ratio = 0.9, raw = True):
        #self.dec_cross_attn_weights = 6x[6x8x900x1450] = layers x (cams x heads x queries x keys)

        all_attn_layers = []
        # loop through layers
        for i in range(self.layers):
            all_attn = []
            # loop through cameras
            for attn in self.dec_cross_attn_weights[i]:
                attn_avg = avg_heads(attn, head_fusion = head_fusion, discard_ratio = discard_ratio)
                if isinstance(target_idx, list):
                    attn_avg = attn_avg[indexes[target_idx]].detach()
                    attn_avg = attn_avg.sum(dim=0)
                else:
                    attn_avg = attn_avg[indexes[target_idx].item()].detach()     
                all_attn.append(attn_avg)
            all_attn_layers.append(all_attn)    
            
        return all_attn_layers
        
    def handle_co_attn_self_query(self, layer):
        attn = self.dec_self_attn_weights[layer]
        grad = self.dec_self_attn_grad[layer]
        attn = avg_heads(attn, grad)
        R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, attn)
        self.R_q_q += R_q_q_add
        self.R_q_i += R_q_i_add

    def handle_co_attn_query(self, layer):
        attn_q_i = self.dec_cross_attn_weights[layer][self.camidx]
        grad_q_i = self.dec_cross_attn_grad[layer][self.camidx]       
        attn_q_i = avg_heads(attn_q_i, grad_q_i)
        self.R_q_i += apply_mm_attention_rules(self.R_q_q, self.R_i_i, attn_q_i,
                                               apply_normalization=self.normalize_self_attention,
                                               apply_self_in_rule_10=self.apply_self_in_rule_10)

    
    # def generate_rollout(self, target_index, indexes, camidx, head_fusion = "min", discard_ratio = 0.9, raw = True):
    #     self.camidx = camidx
    #     self.head_fusion = head_fusion
    #     self.discard_ratio = discard_ratio
        
    #     # initialize relevancy matrices
    #     image_bboxes = self.dec_cross_attn_weights[0].shape[-1]
    #     queries_num = self.dec_self_attn_weights[0].shape[-1]

    #     device = self.dec_cross_attn_weights[0].device
    #     # image self attention matrix
    #     self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(device)
    #     # queries self attention matrix
    #     self.R_q_q = torch.eye(queries_num, queries_num).to(device)

    #     cam_q_i = self.dec_cross_attn_weights[self.layer][self.camidx]
        
    #     cam_q_i = avg_heads(cam_q_i, head_fusion = self.head_fusion, discard_ratio = self.discard_ratio)
        
    #     if raw: 
    #         self.R_q_i = cam_q_i # Only one layer 
    #     else: 
    #         self.R_q_q = compute_rollout_attention(self.dec_self_attn_weights)
    #         #self.R_q_i = torch.matmul(self.R_q_q, torch.matmul(cam_q_i, self.R_i_i))[0]
    #         self.R_q_i = torch.matmul(self.R_q_q, cam_q_i)
            
    #     aggregated = self.R_q_i[indexes[target_index].item()].detach()
                
    #     return aggregated
    
    def generate_rollout(self, target_idx, indexes, camidx, head_fusion = "min", discard_ratio = 0.9, raw = True):
        self.camidx = camidx
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
        # initialize relevancy matrices
        image_bboxes = self.dec_cross_attn_weights[0].shape[-1]
        queries_num = self.dec_self_attn_weights[0].shape[-1]

        device = self.dec_cross_attn_weights[0].device
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)

        cam_q_i = self.dec_cross_attn_weights[self.layer][self.camidx]
        
        cam_q_i = avg_heads(cam_q_i, head_fusion = self.head_fusion, discard_ratio = self.discard_ratio)
        
        if raw: 
            self.R_q_i = cam_q_i # Only one layer 
        else: 
            self.R_q_q = compute_rollout_attention(self.dec_self_attn_weights)
            #self.R_q_i = torch.matmul(self.R_q_q, torch.matmul(cam_q_i, self.R_i_i))[0]
            self.R_q_i = torch.matmul(self.R_q_q, cam_q_i)
        
        if isinstance(target_idx, list):
            aggregated = self.R_q_i[indexes[target_idx]].detach()
            aggregated = aggregated.sum(dim=0)
        else:
            aggregated = self.R_q_i[indexes[target_idx].item()].detach()
                
        return aggregated

    def gradcam(self, cam, grad):
        # FIX
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        grad = grad.mean(dim=0, keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam

    def generate_attn_gradcam(self, target_index, indexes, camidx):
        self.camidx = camidx

        # get cross attn cam from last decoder layer
        cam_q_i = self.dec_cross_attn_weights[-1][self.camidx]
        grad_q_i = self.dec_cross_attn_grads[-1][self.camidx]
        cam_q_i = self.gradcam(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i

        aggregated = self.R_q_i[indexes[target_index].item()].detach()
        return aggregated