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
        self.dec_cross_attn_weights, self.dec_cross_attn_grads, self.dec_self_attn_weights, self.dec_self_attn_grads = [], [], [], []    
    
    def extract_attentions(self, data, target_index=None):
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

    def get_all_attn(self, bbox_idx, indexes, head_fusion="min", discard_ratio=0.9, raw=True):
        #self.dec_cross_attn_weights = 6x[6x8x900x1450] = layers x (cams x heads x queries x keys)
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
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
        # initialize relevancy matrices
        image_bboxes = self.dec_cross_attn_weights[0].shape[-1]
        queries_num = self.dec_self_attn_weights[0].shape[-1]

        device = self.dec_cross_attn_weights[0].device
        # image self attention matrix
        #self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(device)

        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        
        cam_q_i = avg_heads(cam_q_i, head_fusion=self.head_fusion, discard_ratio=self.discard_ratio)
        
        if raw:
            self.R_q_i = cam_q_i # Only one layer
        else: 
            self.R_q_q = compute_rollout_attention(self.dec_self_attn_weights)
            #self.R_q_i = torch.matmul(self.R_q_q, torch.matmul(cam_q_i, self.R_i_i))[0]
            self.R_q_i = torch.matmul(self.R_q_q, cam_q_i)
        
        if isinstance(bbox_idx, list):
            aggregated = self.R_q_i[indexes[bbox_idx]].detach()
            aggregated = aggregated.sum(dim=0)
        else:
            aggregated = self.R_q_i[indexes[bbox_idx].item()].detach()
                
        return aggregated

    def generate_attn_gradcam(self, layer, bbox_idx, indexes, camidx):

        cam_q_i = self.dec_cross_attn_weights[layer][camidx]
        grad_q_i = self.dec_cross_attn_grads[layer][camidx]
        cam_q_i = gradcam(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i

        if isinstance(bbox_idx, list):
            aggregated = self.R_q_i[indexes[bbox_idx]].detach()
            aggregated = aggregated.sum(dim=0)
        else:
            aggregated = self.R_q_i[indexes[bbox_idx].item()].detach()

        return aggregated
    
    def generate_explainability(self, expl_type, layer, bbox_idx, indexes, camidx, head_fusion="min", discard_ratio=0.9, raw=True):
        if expl_type == "Attention Rollout":
            attn = self.generate_rollout(layer, bbox_idx, indexes, camidx, head_fusion, discard_ratio, raw)
        elif expl_type == "Grad-CAM":
            attn = self.generate_attn_gradcam(layer, bbox_idx, indexes, camidx)
        elif expl_type == "Gradient Rollout":
            attn = 0
            # TO-DO
        
        return attn

            

