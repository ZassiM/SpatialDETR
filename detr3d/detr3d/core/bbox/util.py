import torch

def normalize_bbox(bboxes, pc_range):
    """modified version to allow for the mmdet_rc1 cordinate speicifcation
    CAUTION: This requires the bboxes boxes to be adapted as well: box:=[x,y,z,l,w,h, rot_sin, rot_cos, v_x, v_y]
    """

    # bboxes are mmdet_rc1.0 format: x y z l w h rot
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    
    l = bboxes[..., 3:4].log()
    w = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, cz, l, w, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
    else:
        normalized_bboxes = torch.cat(
             (cx, cy, cz, l, w, h, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    """modified version to allow for the mmdet_rc1 cordinate speicifcation
    CAUTION: This requires the normalized boxes to be adapted as well: box:=[x,y,z,l,w,h, rot_sin, rot_cos, v_x, v_y]
    """
    

    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]

    # size
    l = normalized_bboxes[..., 3:4]
    w = normalized_bboxes[..., 4:5]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        # mmdet3d rc1 format lwh instead of wlh
        denormalized_bboxes = torch.cat([cx, cy, cz, l, w, h, rot, vx, vy], dim=-1)
    else:
        # mmdet3d rc1 format lwh instead of wlh
        denormalized_bboxes = torch.cat([cx, cy, cz, l, w, h, rot], dim=-1)
    return denormalized_bboxes
