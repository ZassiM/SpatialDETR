import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
import torchvision.transforms as T
import matplotlib.pyplot as plt


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def evaluate(model, gen, im, device, image_id = None):
    # mean-std normalize the input image (batch-size: 1)
    #img = transform(im).unsqueeze(0).to(device)

    # propagate through the model for obtaining keep indices
    with torch.no_grad():
        output = model(return_loss=False, rescale=True, **im)[0]

    probas = output['pts_bbox']['scores_3d']
    keep = probas>0.7

    # # keep only predictions with 0.7+ confidence
    # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # keep = probas.max(-1).values > 0.9

    # if keep.nonzero().shape[0] <= 1:
    #     return

    # outputs['pred_boxes'] = outputs['pred_boxes'].cpu()

    # # convert boxes from [0; 1] to image scales
    # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # use lists to store the outputs via up-values
    conv_features, dec_self_attn_weights, dec_cross_attn_weights, dec_self_attn_grad, dec_cross_attn_grad = [], [], [], [], []

    hooks = [
        model.module.img_neck.fpn_convs[0].conv.register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),

    ]

    for layer in model.module.pts_bbox_head.transformer.decoder.layers:
        hook = layer.attentions[0].attn.register_forward_hook(
            lambda self, input, output: dec_self_attn_weights.append(output[1])
        )
        hooks.append(hook)
        
    for layer in model.module.pts_bbox_head.transformer.decoder.layers:
        hook = layer.attentions[0].attn.register_backward_hook(
            lambda self, grad_input, grad_output: dec_self_attn_grad.append(grad_input)
        )
        hooks.append(hook)

    for layer in model.module.pts_bbox_head.transformer.decoder.layers:
        hook = layer.attentions[1].attn.register_forward_hook(
            lambda self, input, output: dec_cross_attn_weights.append(output[1])
        )
        hooks.append(hook)

    for layer in model.module.pts_bbox_head.transformer.decoder.layers:
        hook = layer.attentions[1].attn.register_backward_hook(
            lambda self, grad_input, grad_output: dec_cross_attn_grad.append(grad_input)
        )
        hooks.append(hook)

    model.zero_grad()
    outputs = model(return_loss=False, rescale=True, **im)[0]

    outputs = outputs['pts_bbox']['scores_3d']
    one_hot = torch.zeros_like(outputs).to(outputs.device)
    one_hot[0] = 1
    one_hot_vector = one_hot
    one_hot.requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * outputs.cuda())

    one_hot.backward(retain_graph=True)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    dec_self_attn_weights = dec_self_attn_weights[0]
    dec_cross_attn_weights = dec_cross_attn_weights[0]

    # get the feature map shape
    #h, w = conv_features.shape[-2:]
    idx = keep.nonzero()[0] # try first query with score>0.7
    cam = gen.generate_ours(im, idx, dec_self_attn_weights, dec_cross_attn_weights, dec_self_attn_grad, dec_cross_attn_grad, use_lrp=False)

    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     cam = gen.generate_ours(img, idx, use_lrp=False)
    #     cam = (cam - cam.min()) / (cam.max() - cam.min())
    #     cmap = plt.cm.get_cmap('Blues').reversed()
    #     ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap)
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin.detach(), ymin.detach()), xmax.detach() - xmin.detach(), ymax.detach() - ymin.detach(),
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1), attentions[0].size(-2))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-2),attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            #a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        i=0
        for name, module in self.model.named_modules():
            if attention_layer_name in name and "attentions.1" in name and "out_proj" not in name:
                module.register_forward_hook(self.get_attention)
                i+=1    

        self.attentions = []

    def get_attention(self, module, input, output):
        #self.attentions.append(output.cpu())
        self.attentions.append(output[1].cpu())  ##get only attention map (A=Q*K)

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(return_loss=False, rescale=True, **input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)