import numpy as np
import torch
from torch.nn.functional import softmax

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
    _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
    flat[0, indices] = 0
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
        self.dec_cross_attn_weights, self.dec_cross_attn_grad, self.dec_self_attn_weights, self.dec_self_attn_grad = [], [], [], []    
        
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

            
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

    
    
    def generate_rollout(self, target_index, indexes, camidx, head_fusion = "min", discard_ratio = 0.9, raw = True):
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

        self.R_q_q = compute_rollout_attention(self.dec_self_attn_weights)

        cam_q_i = self.dec_cross_attn_weights[-1][self.camidx]
        cam_q_i = avg_heads(cam_q_i, head_fusion = self.head_fusion, discard_ratio = self.discard_ratio)

        if raw: 
            self.R_q_i = cam_q_i # Only last decoder attn 
              
        else: 
            self.R_q_i = torch.matmul(self.R_q_q, torch.matmul(cam_q_i, self.R_i_i))[0]
              
        aggregated = self.R_q_i[indexes[target_index].item()].detach()
                
        return aggregated

    def gradcam(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam

    def generate_attn_gradcam(self, img, target_index, index=None):
        outputs = self.model(img)

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs['pred_logits'])

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)


        # get cross attn cam from last decoder layer
        cam_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn().detach()
        grad_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn_gradients().detach()
        cam_q_i = self.gradcam(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated