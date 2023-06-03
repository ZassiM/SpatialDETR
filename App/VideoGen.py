import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import numpy as np
import cv2
import mmcv

from mmcv.parallel import DataContainer as DC
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from App.App import App

from PIL import Image, ImageTk
from App.UI_baseclass import UI_baseclass


class VideoGen(UI_baseclass):
    '''
    Application User Interface
    '''
    def __init__(self):
        '''
        Tkinter initialization with model loading option.
        '''
        super().__init__()

        # Speeding up the testing
        self.load_from_config()

    def visualize(self):
        if self.video_canvas is None:
            self.video_canvas = tk.Canvas(self)
            self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.canvas_frame = self.video_canvas.create_image(0, 0, anchor='nw', image=None)

        self.select_all_bboxes.set(True)
        self.img_frames, self.img_frames_attention_nobbx, self.og_imgs_frames, self.bbox_coords, self.bbox_cameras, self.bbox_labels, self.all_expl = [], [], [], [], [], [], []

        self.paused = False
        self.video_completed = False

        print("\nGenerating image frames...\n")
        prog_bar = mmcv.ProgressBar(self.video_length)

        for i in range(self.data_idx, self.data_idx + self.video_length):
            self.data_idx = i

            imgs_att, imgs_att_nobbx, imgs_og, bbox_camera, labels = self.generate_video_frame()

            hori = np.concatenate((imgs_att[2], imgs_att[0], imgs_att[1]), axis=1)
            ver = np.concatenate((imgs_att[5], imgs_att[3], imgs_att[4]), axis=1)
            full = np.concatenate((hori, ver), axis=0)

            self.img_frames.append(full)
            self.img_frames_attention_nobbx.append(imgs_att_nobbx)
            self.og_imgs_frames.append(imgs_og)
            self.bbox_cameras.append(bbox_camera)
            self.bbox_labels.append(labels)
            prog_bar.update()

        self.idx_video = 0
        self.data_idx -= (self.video_length - 1)
        
        self.after("idle", self.show_sequence)

    def show_sequence(self):

        if not self.paused and not self.video_completed:
            img_frame = self.img_frames[self.idx_video]

            w, h = self.video_canvas.winfo_width(), self.video_canvas.winfo_height()
            self.img_frame = ImageTk.PhotoImage(Image.fromarray((img_frame * 255).astype(np.uint8)).resize((w, h)))
            self.video_canvas.itemconfig(self.canvas_frame, image=self.img_frame)

            self.idx_video += 1

            if self.idx_video < self.video_length:
                self.after(1, self.show_sequence)
            else:
                self.video_completed = True

    def pause_resume(self):
        if not self.paused and not self.video_completed:
            self.paused = True
        else:
            self.paused = False
            if self.video_completed:
                self.video_completed = False
                self.idx_video = 0

            labels = self.bbox_labels[self.idx_video-1]
            self.update_objects_list(labels)
            self.show_sequence()

    def close_video(self):
        self.destroy()
        self.video_canvas.destroy()
        self.fig.clear()
        self.canvas.draw()

    def show_att_maps_object(self):

        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        if len(self.bbox_idx) == 1:
            self.bbox_idx = self.bbox_idx[0]
            self.fig.clear()
            self.canvas.draw()

            bbox_camera = self.bbox_cameras[self.idx_video-1]

            #{'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5}
            for i, bboxes in enumerate(bbox_camera):
                for b in bboxes:
                    if self.bbox_idx == b[0]:
                        camidx = i
                        bbox_coord = b[1]
                        break

            og_img_frame = self.og_imgs_frames[self.idx_video-1][camidx]
            all_expl = self.img_frames_attention_nobbx[self.idx_video-1]
            labels = self.bbox_labels[self.idx_video-1]

            img_single_obj = og_img_frame[bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]
            attn = all_expl[camidx][bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]

            ax_img = self.fig.add_subplot(self.single_object_spec[0, 0])
            ax_attn = self.fig.add_subplot(self.single_object_spec[0, 1])

            ax_img.imshow(img_single_obj)
            ax_img.set_title(f"{self.class_names[labels[self.bbox_idx]].capitalize()}")
            ax_attn.imshow(attn)
            ax_attn.set_title(f"{self.selected_expl_type.get()}")

            ax_img.axis('off')
            ax_attn.axis("off")
            
            self.fig.tight_layout(pad=0)
            self.canvas.draw()
        
        self.after(1, self.show_att_maps_object)

    def generate_video_frame(self):

        self.update_data()

        # Avoid selecting all layers and all cameras. Only the last layer will be visualized
        if self.show_all_layers.get() and self.selected_camera.get() == -1:
            self.selected_layer.set(self.Attention.layers - 1)

        # Extract the selected bounding box indexes from the menu
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        self.attn_list = self.Attention.generate_explainability_cameras(self.selected_expl_type.get(), self.selected_layer.get(), self.bbox_idx, self.nms_idxs, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get())

        # Generate images list with bboxes on it
        cam_imgs, og_imgs, bbox_cameras, att_nobbx = [], [], [], []  # Used for video generation

        for camidx in range(len(self.imgs)):
            og_img = cv2.cvtColor(self.imgs[camidx], cv2.COLOR_BGR2RGB)
            og_imgs.append(og_img)

            attn_img = self.imgs[camidx].astype(np.uint8)
            attn = self.attn_list[camidx]

            attn_img = self.overlay_attention_on_image(attn_img, attn)      
            attn_img = cv2.cvtColor(attn_img, cv2.COLOR_BGR2RGB)
            att_nobbx.append(attn_img)

            img, bbox_camera = draw_lidar_bbox3d_on_img(
                    self.pred_bboxes,
                    self.imgs[camidx],
                    self.img_metas['lidar2img'][camidx],
                    self.img_metas,
                    color=(0, 255, 0),
                    with_label=self.show_labels.get(),
                    all_bbx=self.BB_bool.get(),
                    bbx_idx=self.bbox_idx,
                    mode_2d=self.bbox_2d.get(),
                    camidx=camidx)  
            
            bbox_cameras.append(bbox_camera)       

            attn = self.attn_list[camidx]
            img = self.overlay_attention_on_image(img, attn)   

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cam_imgs.append(img)

        return cam_imgs, att_nobbx, og_imgs, bbox_cameras, self.labels

    def update_data(self):
        '''
        Predict bboxes and extracts attentions.
        '''
        # Load selected data from dataloader, manual DataContainer fixes are needed
        data = self.dataloader.dataset[self.data_idx]
        metas = [[data['img_metas'][0].data]]
        img = [data['img'][0].data.unsqueeze(0)]
        data['img_metas'][0] = DC(metas, cpu_only=True)
        data['img'][0] = DC(img)
        self.data = data

        # Attention scores are extracted, together with gradients if grad-CAM is selected
        if self.selected_expl_type.get() not in ["Grad-CAM", "Gradient Rollout"]:
            outputs = self.Attention.extract_attentions(self.data)
        else:
            outputs = self.Attention.extract_attentions(self.data, self.bbox_idx)

        # Those are needed to index the bboxes decoded by the NMS-Free decoder
        self.nms_idxs = self.model.module.pts_bbox_head.bbox_coder.get_indexes()

        # Extract predicted bboxes and their labels
        self.outputs = outputs[0]["pts_bbox"]
        self.thr_idxs = self.outputs['scores_3d'] > self.selected_threshold.get()

        self.pred_bboxes = self.outputs["boxes_3d"][self.thr_idxs]
        self.pred_bboxes.tensor.detach()
        self.labels = self.outputs['labels_3d'][self.thr_idxs]

        # Extract image metas which contain, for example, the lidar to camera projection matrices
        self.img_metas = self.data["img_metas"][0]._data[0][0]
        ori_shape = self.img_metas["ori_shape"] # Used for removing the padded pixels

        # Extract the 6 camera images from the data and remove the padded pixels
        imgs = self.data["img"][0]._data[0].numpy()[0]
        imgs = imgs.transpose(0, 2, 3, 1)[:, :ori_shape[0], :ori_shape[1], :] # [num_cams x height x width x channels]
        
        # Denormalize the images
        mean = np.array(self.img_norm_cfg["mean"], dtype=np.float32)
        std = np.array(self.img_norm_cfg["std"], dtype=np.float32)

        for i in range(len(imgs)):
            imgs[i] = mmcv.imdenormalize(imgs[i], mean, std, to_bgr=False)
        self.imgs = imgs.astype(np.uint8)

        # Update the Bounding box menu with the predicted labels
        if self.old_data_idx != self.data_idx or self.old_thr != self.selected_threshold.get() or self.new_model:
            self.update_objects_list()
            self.initialize_bboxes()




