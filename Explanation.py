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

def avg_heads(attn):
    attn = attn.reshape(-1, attn.shape[-2], attn.shape[-1])
    attn = attn.clamp(min=0).mean(dim=0)
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

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

            
# self.dec_self_attn_weights, self.dec_cross_attn_weights, self.dec_self_attn_grad, self.dec_cross_attn_grad = [], [], [], []

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

    def generate_ours(self, data, target_index, indexes, camidx, use_lrp=False, normalize_self_attention=True, apply_self_in_rule_10=True):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10
        self.camidx = camidx
        dec_self_attn_weights, dec_cross_attn_weights, dec_self_attn_grad, dec_cross_attn_grad = [], [], [], []
        
        hooks = []
        for layer in self.model.module.pts_bbox_head.transformer.decoder.layers:
            hooks.append(
            layer.attentions[0].attn.register_forward_hook(
                lambda self, input, output: dec_self_attn_weights.append(output[1])
            ))
            hooks.append(
            layer.attentions[1].attn.register_forward_hook(
                lambda self, input, output: dec_cross_attn_weights.append(output[1])
            ))
            

        # self: 900x900, cross: 6x8x900x1450
        # I have to select one attnera, and backpropagate one class
        outputs = self.model(return_loss=False, rescale=True, **data)
        outputs = outputs[0]["pts_bbox"]['scores_3d']

        one_hot = torch.zeros_like(outputs).to(outputs.device)
        one_hot[target_index] = 1
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot * outputs)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        for layer in self.model.module.pts_bbox_head.transformer.decoder.layers:
            dec_self_attn_grad.append(layer.attentions[0].attn.get_attn_gradients())
            dec_cross_attn_grad.append(layer.attentions[1].attn.get_attn_gradients())
        
        for hook in hooks:
            hook.remove()
        
        self.dec_self_attn_weights, self.dec_cross_attn_weights, self.dec_self_attn_grad, self.dec_cross_attn_grad = \
            dec_self_attn_weights, dec_cross_attn_weights, dec_self_attn_grad, dec_cross_attn_grad
        
        H, Q, K = self.dec_cross_attn_grad[0].shape
        CAMS = 6
        K_NEW = int(K/CAMS)
        n_layers = len(self.dec_cross_attn_weights)
        for i in range(n_layers):
            self.dec_cross_attn_weights[i] = self.dec_cross_attn_weights[i].reshape(CAMS, H, Q, K_NEW)
            self.dec_cross_attn_grad[i] = self.dec_cross_attn_grad[i].reshape(CAMS, H, Q, K_NEW)
        
        # initialize relevancy matrices
        image_bboxes = self.dec_cross_attn_weights[0].shape[-1]
        queries_num = self.dec_self_attn_weights[0].shape[-1]

        device = self.dec_cross_attn_weights[0].device
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(device)


        # decoder self attention of queries followd by multi-modal attention
        for layer in range(n_layers):
            # decoder self attention
            self.handle_co_attn_self_query(layer)

            # encoder decoder attention
            self.handle_co_attn_query(layer)

        aggregated = self.R_q_i[indexes[target_index], :].detach()
        return aggregated
    
    def generate_rollout(self, data, target_index, indexes, camidx, head_fusion = "min"):

        self.camidx = camidx
        self.head_fusion = head_fusion
        dec_self_attn_weights, dec_cross_attn_weights = [], []
        
        hooks = []
        for layer in self.model.module.pts_bbox_head.transformer.decoder.layers:
            hooks.append(
            layer.attentions[0].attn.register_forward_hook(
                lambda self, input, output: dec_self_attn_weights.append(output[1])
            ))
            hooks.append(
            layer.attentions[1].attn.register_forward_hook(
                lambda self, input, output: dec_cross_attn_weights.append(output[1])
            ))
            
        # self: 900x900, cross: 6x8x900x1450
        # I have to select one attnera, and backpropagate one class
        outputs = self.model(return_loss=False, rescale=True, **data)
        
        for hook in hooks:
            hook.remove()
        
        for i in range(len(dec_self_attn_weights)):
            if head_fusion == "mean":
                dec_self_attn_weights[i] = dec_self_attn_weights[i].detach().mean(dim=0)
            elif head_fusion == "max":
                dec_self_attn_weights[i] = dec_self_attn_weights[i].detach().max(dim=0)[0]
            elif head_fusion == "min":
                dec_self_attn_weights[i] = dec_self_attn_weights[i].detach().min(dim=0)[0]
            

        
        self.dec_self_attn_weights, self.dec_cross_attn_weights,  = \
            dec_self_attn_weights, dec_cross_attn_weights,
        
        # H, Q, K = self.dec_cross_attn_weights[0].shape
        # CAMS = 6
        # K_NEW = int(K/CAMS)
        # n_layers = len(self.dec_cross_attn_weights)
        
        # for i in range(n_layers):
        #     self.dec_cross_attn_weights[i] = self.dec_cross_attn_weights[i].reshape(CAMS, H, Q, K_NEW)
        
        # initialize relevancy matrices
        image_bboxes = self.dec_cross_attn_weights[0].shape[-1]
        queries_num = self.dec_self_attn_weights[0].shape[-1]

        device = self.dec_cross_attn_weights[0].device
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)


        self.R_q_q = compute_rollout_attention(self.dec_self_attn_weights)

        cam_q_i = self.dec_cross_attn_weights[-1][camidx]
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        if head_fusion == "mean":
            cam_q_i = cam_q_i.mean(dim=0)
        elif head_fusion == "max":
            cam_q_i = cam_q_i.max(dim=0)[0]
        elif head_fusion == "min":
            cam_q_i = cam_q_i.min(dim=0)[0]
        #self.R_q_i = torch.matmul(self.R_q_q.t(), torch.matmul(cam_q_i, self.R_i_i))
        self.R_q_i = cam_q_i
        aggregated = self.R_q_i[indexes[target_index].item(), :].detach()
        return aggregated