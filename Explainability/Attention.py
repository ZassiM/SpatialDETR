import torch

def avg_heads(attn, head_fusion="min", discard_ratio=0.9):
    if head_fusion == "mean":
        attn = attn.mean(dim=0)
    elif head_fusion == "max":
        attn = attn.max(dim=0)[0]
    elif head_fusion == "min":
        attn = attn.min(dim=0)[0]

    flat = attn.view(attn.size(0), -1)
    _, indices = flat.topk(int(attn.size(-1)*discard_ratio), -1, False)
    for i in range(len(indices)):
        flat[i, indices[i]] = 0
    return attn

# rule 5 from paper
def avg_heads_og(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rules 6 + 7 from paper
def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition

# rule 10 from paper
def apply_mm_attention_rules(R_ss, cam_sq, handle_residual_bool=True, apply_rule=True):
    R_ss_normalized = R_ss
    if handle_residual_bool:
        R_ss_normalized = handle_residual(R_ss)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), cam_sq)
    if not apply_rule:
        R_sq_addition = cam_sq
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
            
    def handle_co_attn_self_query(self, layer):
        # grad = self.dec_self_attn_grads[layer]
        cam = self.dec_self_attn_weights[layer]
        # cam = avg_heads_og(cam, grad)
        R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, cam)
        self.R_q_q += R_q_q_add
        self.R_q_i += R_q_i_add

    def handle_co_attn_query(self, layer, camidx, handle_residual, apply_rule):
        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = avg_heads_og(cam_q_i, grad_q_i)
        self.R_q_i += apply_mm_attention_rules(self.R_q_q, cam_q_i, handle_residual, apply_rule)
    
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

        if target_index is None:
            with torch.no_grad():
                outputs = self.model(return_loss=False, rescale=True, **data)
        
        else:
            for layer in self.model.module.pts_bbox_head.transformer.decoder.layers:
                hooks.append(
                layer.attentions[0].attn.register_backward_hook(
                    lambda _, grad_input, grad_output: self.dec_self_attn_grads.append(grad_input[0].permute(1,0,2)[0])
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
        
        for hook in hooks:
            hook.remove()
        
        return outputs

    def get_all_attn(self, bbox_idx, indexes, head_fusion="min", discard_ratio=0.9, raw=True):
        # self.dec_cross_attn_weights = 6x[6x8x900x1450] = layers x (cams x heads x queries x keys)
        all_attn_layers = []
        # loop through layers
        for i in range(self.layers):
            all_attn = []
            # loop through cameras
            for attn in self.dec_cross_attn_weights[i]:
                attn_avg = avg_heads(attn, head_fusion=head_fusion, discard_ratio=discard_ratio)
                if isinstance(bbox_idx, list):
                    attn_avg = attn_avg[indexes[bbox_idx]].detach()
                    attn_avg = attn_avg.sum(dim=0)
                else:
                    attn_avg = attn_avg[indexes[bbox_idx].item()].detach()     
                all_attn.append(attn_avg)
            all_attn_layers.append(all_attn)    
            
        return all_attn_layers
        
    def generate_rollout(self, layer, bbox_idx, indexes, camidx, head_fusion="min", discard_ratio=0.9, raw=True):  
        ''' Generates Attention Rollout for XAI. '''      

        # initialize relevancy matrices
        queries_num = self.dec_self_attn_weights[0].shape[-1]
        device = self.dec_cross_attn_weights[0].device

        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)

        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        cam_q_i = avg_heads(cam_q_i, head_fusion, discard_ratio)
        
        if raw:
            self.R_q_i = cam_q_i # Only one layer
        else: 
            self.R_q_q = compute_rollout_attention(self.dec_self_attn_weights)
            self.R_q_i = torch.matmul(self.R_q_q, cam_q_i)
        
        if isinstance(bbox_idx, list):
            attention_map = self.R_q_i[indexes[bbox_idx]].detach()
            attention_map = attention_map.sum(dim=0)
        else:
            attention_map = self.R_q_i[indexes[bbox_idx].item()].detach()
                
        return attention_map

    def generate_attn_gradcam(self, layer, bbox_idx, indexes, camidx):
        ''' Generates Grad-CAM for XAI. '''      

        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = gradcam(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i

        if isinstance(bbox_idx, list):
            attention_map = self.R_q_i[indexes[bbox_idx]].detach()
            attention_map = attention_map.sum(dim=0)
        else:
            attention_map = self.R_q_i[indexes[bbox_idx].item()].detach()

        return attention_map
    
    def generate_grad_roll(self, layer, bbox_idx, indexes, camidx, handle_residual, apply_rule):
        # initialize relevancy matrices
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
            
        if isinstance(bbox_idx, list):
            attention_map = self.R_q_i[indexes[bbox_idx]].detach()
            attention_map = attention_map.sum(dim=0)
        else:
            attention_map = self.R_q_i[indexes[bbox_idx].item()].detach()
                
        return attention_map
    
    def generate_explainability(self, expl_type, layer, bbox_idx, indexes, camidx, head_fusion="min", discard_ratio=0.9, raw=True, handle_residual=True, apply_rule=True):
        if expl_type == "Attention Rollout":
            attn = self.generate_rollout(layer, bbox_idx, indexes, camidx, head_fusion, discard_ratio, raw)
        elif expl_type == "Grad-CAM":
            attn = self.generate_attn_gradcam(layer, bbox_idx, indexes, camidx)
        elif expl_type == "Partial-LRP":
            attn = 0
            # TO-DO
        elif expl_type == "Gradient Rollout":
            attn = self.generate_grad_roll(layer, bbox_idx, indexes, camidx, handle_residual, apply_rule)
            # TO-DO
        
        return attn

            

