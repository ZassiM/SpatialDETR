import torch


def avg_heads(attn, head_fusion="min", discard_ratio=0.9):
    if head_fusion == "mean":
        attn = attn.mean(dim=0)
    elif head_fusion == "max":
        attn = attn.max(dim=0)[0]
    elif head_fusion == "min":
        attn = attn.min(dim=0)[0]

    # flat = attn.view(-1)
    # _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
    # indices = indices[indices != 0]
    # flat[indices] = 0
    return attn


def avg_heads_og(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


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


def gradcam(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    grad = grad.mean(dim=0, keepdim=True)
    cam = (cam * grad).mean(0).clamp(min=0)
    return cam


class Attention:
    '''
    Generates attention maps and different explainability features.
    '''
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.layers = 0
        for _ in self.model.module.pts_bbox_head.transformer.decoder.layers:
            self.layers += 1
        self.height_feats, self.width_feats = None, None
            
    def handle_co_attn_self_query(self, layer):
        cam = self.dec_self_attn_weights[layer]
        self.apply_self_attention_rules(self.R_q_q, self.R_q_i, cam)

    def handle_co_attn_query(self, layer, camidx, handle_residual, apply_rule):
        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = avg_heads_og(cam_q_i, grad_q_i)
        self.apply_mm_attention_rules(self.R_q_q, cam_q_i, handle_residual, apply_rule)

    def apply_self_attention_rules(self, R_qq, R_qi, cam_qq):
        self.R_q_i += torch.matmul(cam_qq, R_qi)
        self.R_q_q += torch.matmul(cam_qq, R_qq)

    def apply_mm_attention_rules(self, R_qq, cam_qi, handle_residual_bool=True, apply_rule=True):
        R_qq_normalized = R_qq
        if handle_residual_bool:
            R_qq_normalized = handle_residual(R_qq)
        R_qi_addition = torch.matmul(R_qq_normalized.t(), cam_qi)
        if not apply_rule:
            R_qi_addition = cam_qi
        R_qi_addition[torch.isnan(R_qi_addition)] = 0
        self.R_q_i += R_qi_addition
    
    def extract_attentions(self, data, target_index=None):
        self.dec_cross_attn_weights, self.dec_cross_attn_grads, self.dec_self_attn_weights, self.dec_self_attn_grads = [], [], [], [] 

        hooks = []
        for layer in self.model.module.pts_bbox_head.transformer.decoder.layers:
            hooks.append(
                layer.attentions[0].attn.register_forward_hook(
                    lambda _, input, output: self.dec_self_attn_weights.append(output[1][0])
                ))
            hooks.append(
                layer.attentions[1].attn.register_forward_hook(
                    lambda _, input, output: self.dec_cross_attn_weights.append(output[1])
                ))
        
        if self.height_feats is None:
            conv_feats = []
            self.model.module.img_backbone.register_forward_hook(
                lambda _, input, output: conv_feats.append(output)
            )

        if target_index is None:
            with torch.no_grad():
                outputs = self.model(return_loss=False, rescale=True, **data)

        else:
            for layer in self.model.module.pts_bbox_head.transformer.decoder.layers:
                hooks.append(
                    layer.attentions[0].attn.register_backward_hook(
                        lambda _, grad_input, grad_output: self.dec_self_attn_grads.append(grad_input[0].permute(1, 0, 2)[0])
                    ))
     
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

        if self.height_feats is None:
            self.height_feats, self.width_feats = conv_feats[0][0].shape[-2:]
                                                                     
        for hook in hooks:
            hook.remove()
        
        return outputs
        
    def generate_rollout(self, layer, camidx, head_fusion="min", discard_ratio=0.9, raw=True):  
        ''' Generates Attention Rollout for XAI. '''      

        # initialize relevancy matrices
        queries_num = self.dec_self_attn_weights[0].shape[-1]
        device = self.dec_cross_attn_weights[0].device

        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)

        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        cam_q_i = avg_heads(cam_q_i, head_fusion, discard_ratio)
        
        if raw:
            self.R_q_i = cam_q_i  # Only one layer
        else:
            self.R_q_q = compute_rollout_attention(self.dec_self_attn_weights)
            self.R_q_i = torch.matmul(self.R_q_q, cam_q_i)

    def generate_gradcam(self, layer, camidx):
        ''' Generates Grad-CAM for XAI. '''      

        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = gradcam(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
    
    def generate_gradroll(self, layer, camidx, handle_residual, apply_rule):
        # initialize relevancy matrices

        #self.extract_attentions(bbox_idx)
        queries_num = self.dec_self_attn_weights[0].shape[-1]
        image_bboxes = self.dec_cross_attn_weights[0].shape[-1]
        
        device = self.dec_cross_attn_weights[0].device

        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(device)

        # decoder self attention of queries followd by multi-modal attention
        for layer in range(self.layers):
            # decoder self attention
            self.handle_co_attn_self_query(layer)

            # encoder decoder attention
            self.handle_co_attn_query(layer, camidx, handle_residual, apply_rule)
    
    def generate_explainability_cameras(self, expl_type, layer, bbox_idx, indexes, head_fusion="min", discard_ratio=0.9, raw=True, handle_residual=True, apply_rule=True):
        
        attention_maps = []
        for camidx in range(6):
            if expl_type == "Attention Rollout":
                self.generate_rollout(layer, camidx, head_fusion, discard_ratio, raw)
            elif expl_type == "Grad-CAM":
                self.generate_gradcam(layer, camidx)
            elif expl_type == "Gradient Rollout":
                self.generate_gradroll(layer, camidx, handle_residual, apply_rule)
            elif expl_type == "Partial-LRP":
                attention_maps = 0
            attention_maps.append(self.R_q_i[indexes[bbox_idx]].detach().cpu())

        # num_cams x num_objects x 1450
            
        attention_maps = torch.stack(attention_maps)
        attention_maps = attention_maps.permute(1, 0, 2) # num_objects x num_cams x 1450 # take only the selected objects

        # normalize across cameras
        for i in range(len(attention_maps)):
            attention_maps[i] = (attention_maps[i] - attention_maps[i].min()) / (attention_maps[i].max() - attention_maps[i].min())

        # now attention maps can be overlayed
        attention_maps = attention_maps.max(dim=0)[0]  # num_cams x [1450]

        attention_maps = self.interpolate_expl(attention_maps)

        return attention_maps
    
    def generate_explainability_layers(self, expl_type, camidx, bbox_idx, indexes, head_fusion="min", discard_ratio=0.9, raw=True, handle_residual=True, apply_rule=True):
        
        attention_maps = []
        for layer in range(6):
            if expl_type == "Attention Rollout":
                self.generate_rollout(layer, camidx, head_fusion, discard_ratio, raw)
            elif expl_type == "Grad-CAM":
                self.generate_gradcam(layer, camidx)
            elif expl_type == "Gradient Rollout":
                self.generate_gradroll(layer, camidx, handle_residual, apply_rule)
            elif expl_type == "Partial-LRP":
                attention_maps = 0
            attention_maps.append(self.R_q_i[indexes[bbox_idx]].detach().cpu())

        # num_layers x num_objects x 1450
        attention_maps = torch.stack(attention_maps)
        attention_maps = attention_maps.permute(1, 0, 2) # num_objects x num_layers x 1450 # take only the selected objects

        # now attention maps can be overlayed
        attention_maps = attention_maps.max(dim=0)[0]  # num_layers x [1450]

        for i in range(len(attention_maps)):
            attention_maps[i] = (attention_maps[i] - attention_maps[i].min()) / (attention_maps[i].max() - attention_maps[i].min())

        attention_maps = self.interpolate_expl(attention_maps)

        return attention_maps

    def interpolate_expl(self, attention_maps):
        attention_maps_inter = []
        if attention_maps.dim() == 1:
            attention_maps.unsqueeze_(0)
        for i in range(len(attention_maps)):
            attn = attention_maps[i].view(1, 1, self.height_feats, self.width_feats)
            attn = torch.nn.functional.interpolate(attn, scale_factor=8, mode='bilinear')
            attn = attn.view(attn.shape[2], attn.shape[3])
            attention_maps_inter.append(attn)

        attention_maps_inter = torch.stack(attention_maps_inter)

        return attention_maps_inter