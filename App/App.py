import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import numpy as np
import cv2
import mmcv

from mmcv.parallel import DataContainer as DC
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from App.UI_baseclass import UI_baseclass

from PIL import Image, ImageTk
import pickle
from tkinter import filedialog as fd


class App(UI_baseclass):
    '''
    Application User Interface
    '''
    def __init__(self):
        '''
        Tkinter initialization with model loading option.
        '''
        super().__init__()

        # Speeding up the testing
        #self.load_from_config()

    def evaluate_expl(self):
        print(f"Evaluating {self.selected_expl_type.get()}...")

        bbox_idx = [0]
        initial_idx = 500
        evaluation_lenght = 20
        num_tokens = int(0.25 * 1450)
        outputs_pert = []
        dataset = self.dataloader.dataset
        prog_bar = mmcv.ProgressBar(evaluation_lenght)

        for i in range(initial_idx, initial_idx + evaluation_lenght):
            data = self.dataloader.dataset[i]
            metas = [[data['img_metas'][0].data]]
            img = [data['img'][0].data.unsqueeze(0)] # img[0] = torch.Size([1, 6, 3, 928, 1600])
            data['img_metas'][0] = DC(metas, cpu_only=True)
            data['img'][0] = DC(img)

            # Attention scores are extracted, together with gradients if grad-CAM is selected
            if self.selected_expl_type.get() not in ["Grad-CAM", "Gradient Rollout"]:
                self.Attention.extract_attentions(data)
            else:
                self.Attention.extract_attentions(data, bbox_idx)

            nms_idxs = self.model.module.pts_bbox_head.bbox_coder.get_indexes()

            attn_list = []
            topk_list = []
            
            for camidx in range(6):
                attn = self.Attention.generate_explainability(self.selected_expl_type.get(), self.selected_layer.get(), bbox_idx, nms_idxs, camidx, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get())
                attn = attn.view(1, 1, 29, 50)
                attn = torch.nn.functional.interpolate(attn, scale_factor=32, mode='bilinear')
                attn = attn.view(attn.shape[2], attn.shape[3]).cpu()
                attn_list.append(attn)

            attn_max = np.max(np.concatenate(attn_list))
            for i in range(len(attn_list)):
                attn = attn_list[i]
                attn /= attn_max
                _, indices = torch.topk(attn.flatten(), k=num_tokens)
                indices = np.array(np.unravel_index(indices.numpy(), attn.shape)).T
                topk_list.append(indices)

            img_og_list = [] # list of original images
            img_pert_list = [] # list of perturbed images

            # Denormalization is needed, because data is normalized 
            img = img[0][0]
            for i in range(len(img)):
                img_og = img[i].permute(1, 2, 0)

                img_og_list.append(img_og)
                img_pert = img_og.clone()
                # Image perturbation by setting pixels to (0,0,0)
                for idx in topk_list[i]:
                    img_pert[idx[0], idx[1]] = 0
                img_pert_list.append(img_pert.permute(2, 0, 1))

            # Save the perturbed 6 camera images into the data input
            img = [torch.stack((img_pert_list)).unsqueeze(0)]  # img[0] = torch.Size([1, 6, 3, 928, 1600])
            data['img'][0] = DC(img)

            # Second forward
            with torch.no_grad():
                output = self.model(return_loss=False, rescale=True, **data)

            outputs_pert.extend(output)
            torch.cuda.empty_cache()

            prog_bar.update()
        
        print("\nCompleted.\n")

        kwargs = {}
        eval_kwargs = self.cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric="bbox", **kwargs))
        print(dataset.evaluate(outputs_pert, **eval_kwargs))

    def show_attention_maps(self, grid_clm=1):
        '''
        Shows the attention map for explainability.
        '''
        # If we want to visualize all layers or all cameras:
        if self.show_all_layers.get() or self.selected_camera.get() == -1:
            # Select the center of the grid to plot the attentions and add 2x2 subgrid
            layer_grid = self.spec[1, grid_clm].subgridspec(2, 3)
            fontsize = 8
        else:
            fontsize = 12
            
        for i in range(len(self.attn_list)):

            if self.show_all_layers.get() or self.selected_camera.get() == -1:
                ax_attn = self.fig.add_subplot(layer_grid[i > 2, i if i < 3 else i - 3])
            else:
                ax_attn = self.fig.add_subplot(self.spec[1, grid_clm])
            
            if self.show_all_layers.get():
                attn = self.attn_list[i]
            elif self.selected_camera.get() == -1:
                attn = self.attn_list[self.cam_idx[i]]
            else:
                attn = self.attn_list[self.selected_camera.get()]
                
            ax_attn.axis('off')
            ax_attn.imshow(attn, vmin=0, vmax=1)            

            # Set title accordinly
            if self.show_all_layers.get():
                title = f'{list(self.cameras.keys())[self.selected_camera.get()]}, layer {i}'
            elif self.selected_camera.get() == -1:
                title = f'{list(self.cameras.keys())[self.cam_idx[i]]}, layer {self.selected_layer.get()}'
            else:
                title = None

            # If doing Attention Rollout, visualize head fusion type
            if self.show_all_layers.get() or self.selected_camera.get() == -1 and self.selected_expl_type.get() == "Attention Rollout":
                title += f', {self.selected_head_fusion.get()}'

            # Show attention camera contributon for one object
            if self.attn_contr.get() and self.selected_camera.get() == -1:
                self.update_scores()
                score = self.scores_perc[self.cam_idx[i]]
                title += f', {score}%'
                ax_attn.axhline(y=0, color='black', linewidth=10)
                ax_attn.axhline(y=0, color='green', linewidth=10, xmax=score/100)

            self.fig.tight_layout(pad=0)
            ax_attn.set_title(title, fontsize=fontsize)

    def visualize(self):
        '''
        Visualizes predicted bounding boxes on all the cameras and shows
        the attention map in the middle of the plot.
        '''
        if self.canvas is None or self.video_gen_bool:
            # Create canvas with the figure embedded in it, and update it after each visualization
            if self.video_gen_bool:
                self.canvas.pack_forget()
                self.menubar.delete('end', 'end')
            self.fig = plt.figure()
            self.spec = self.fig.add_gridspec(3, 3)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.video_gen_bool = False

        self.fig.clear()

        # Data is updated only when data idx, prediction threshold or the model is changed
        if self.old_data_idx != self.data_idx or self.old_thr != self.selected_threshold.get() or self.old_expl_type != self.selected_expl_type.get() or self.new_model:
            print("\nDetecting bounding boxes...")
            self.update_data()
            if self.new_model:
                self.new_model = False
            if self.old_thr != self.selected_threshold.get():
                self.old_thr = self.selected_threshold.get()
            if self.old_data_idx != self.data_idx:
                self.old_data_idx = self.data_idx
            if self.old_expl_type != self.selected_expl_type.get():
                self.old_expl_type = self.selected_expl_type.get()

        # Avoid selecting all layers and all cameras. Only the last layer will be visualized
        if self.show_all_layers.get() and self.selected_camera.get() == -1:
            self.selected_layer.set(self.Attention.layers - 1)

        # Extract the selected bounding box indexes from the menu
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        if self.selected_expl_type.get() == "Gradient Rollout":
            self.update_data()
            self.show_all_layers.set(False)

        print("Generating attention maps...")
        # Explainable attention maps generation
        if self.show_all_layers.get():
            self.attn_list = self.Attention.generate_explainability_layers(self.selected_expl_type.get(), self.selected_camera.get(), self.bbox_idx, self.nms_idxs, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get())
        else:
            self.attn_list = self.Attention.generate_explainability_cameras(self.selected_expl_type.get(), self.selected_layer.get(), self.bbox_idx, self.nms_idxs, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get())

        self.show_attention_maps()

        # Extract Ground Truth bboxes if wanted
        if self.GT_bool.get():
            self.gt_bbox = self.dataloader.dataset.get_ann_info(self.data_idx)['gt_bboxes_3d']
        else:
            self.gt_bbox = None

        # Generate images list with bboxes on it
        print("Generating camera images...")
        self.cam_imgs = []
        for camidx in range(len(self.imgs)):
            img, _ = draw_lidar_bbox3d_on_img(
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
            
            if self.GT_bool.get():
                img, _ = draw_lidar_bbox3d_on_img(
                        self.gt_bbox,
                        img,
                        self.img_metas['lidar2img'][camidx],
                        self.img_metas,
                        color=(255, 0, 0),
                        mode_2d=self.bbox_2d.get())
                
            if self.overlay_bool.get() and ((self.selected_camera.get() != -1 and camidx == self.selected_camera.get()) or (self.selected_camera.get() == -1)):
                if self.show_all_layers.get():
                    attn = self.attn_list[self.selected_layer.get()]
                elif self.selected_camera.get() == -1:
                    attn = self.attn_list[camidx]
                else:
                    attn = self.attn_list[self.selected_camera.get()]

                img = self.overlay_attention_on_image(img, attn)            

            # num_tokens = int(1450)
            # _, indices = torch.topk(torch.from_numpy(attn).flatten(), k=num_tokens)
            # indices = np.array(np.unravel_index(indices.numpy(), attn.shape)).T
            # for idx in indices:
            #     img[idx[0], idx[1]] = 0

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.cam_imgs.append(img)

        print("Done.\n")

        # Visualize the generated images list on the figure subplots
        for i in range(len(self.imgs)):
            if i < 3:
                ax = self.fig.add_subplot(self.spec[0, i])
            else:
                ax = self.fig.add_subplot(self.spec[2, i-3])
            
            ax.imshow(self.cam_imgs[self.cam_idx[i]])
            ax.axis('off')
        
        self.fig.tight_layout(pad=0)
        self.canvas.draw()

        # take a screenshot if the option is selected
        if self.capture_bool.get():
            self.capture()

    def show_video(self):

        if self.canvas and not self.video_gen_bool or not self.canvas:
            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.menubar.add_command(label="Pause/Resume", command=self.pause_resume)
            self.canvas = tk.Canvas(self)
            self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.canvas_frame = self.canvas.create_image(0, 0, anchor='nw', image=None)
            self.video_gen_bool = True

        if self.single_object_window is None:
            self.single_object_window = tk.Toplevel(self)
            self.single_object_window.title("Object Explainability Visualizer")
            self.fig_obj = plt.figure()
            self.single_object_spec = self.fig_obj.add_gridspec(3, 2)
            self.obj_canvas = FigureCanvasTkAgg(self.fig_obj, master=self.single_object_window)
            self.obj_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.idx_video = 0
        self.paused = False

        if not hasattr(self, "img_frames"):
            self.generate_video()
        
        self.old_bbox_idx = None
        self.show_sequence()

    def generate_video(self):

        self.select_all_bboxes.set(True)
        self.img_frames, self.img_frames_attention_nobbx, self.og_imgs_frames, self.bbox_coords, self.bbox_cameras, self.bbox_labels, self.all_expl = [], [], [], [], [], [], []

        print("\nGenerating image frames...\n")
        prog_bar = mmcv.ProgressBar(self.video_length)
        for i in range(self.data_idx, self.data_idx + self.video_length):
            self.data_idx = i

            img_att_expl, img_att_nobbx = [], []
            for expl in self.expl_types:
                self.selected_expl_type.set(expl)
                imgs_att, imgs_att_nobbx, imgs_og, bbox_camera, labels = self.generate_video_frame()
                img_att_expl.append(imgs_att)
                img_att_nobbx.append(imgs_att_nobbx)

            img_att = img_att_expl[0]
            hori = np.concatenate((img_att[2], img_att[0], img_att[1]), axis=1)
            ver = np.concatenate((img_att[5], img_att[3], img_att[4]), axis=1)
            full = np.concatenate((hori, ver), axis=0)

            self.img_frames.append(full)
            self.img_frames_attention_nobbx.append(img_att_nobbx)
            self.og_imgs_frames.append(imgs_og)
            self.bbox_cameras.append(bbox_camera)
            self.bbox_labels.append(labels)

            prog_bar.update()

        self.data_idx -= (self.video_length - 1)
        print("\nVideo generated.\n")

    def show_sequence(self):
        if not self.paused:
            img_frame = self.img_frames[self.idx_video]
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            self.img_frame = ImageTk.PhotoImage(Image.fromarray((img_frame * 255).astype(np.uint8)).resize((w, h)))
            self.canvas.itemconfig(self.canvas_frame, image=self.img_frame)
            self.idx_video += 1
            self.update_info_label(idx=self.data_idx + self.idx_video)

            if self.idx_video >= self.video_length:
                self.idx_video = 0

            self.after_seq_id = self.after(self.frame_rate, self.show_sequence)

    def pause_resume(self):
        if not self.paused:
            self.paused = True
            self.after_cancel(self.after_seq_id)
            labels = self.bbox_labels[self.idx_video-1]
            self.update_objects_list(labels)
            self.show_att_maps_object()
        else:
            self.paused = False
            self.after_cancel(self.after_obj_id)
            self.show_sequence()

    def show_att_maps_object(self):
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        if len(self.bbox_idx) == 1 and self.bbox_idx != self.old_bbox_idx:
            self.old_bbox_idx = self.bbox_idx
            self.fig_obj.clear()
            self.bbox_idx = self.bbox_idx[0]
            bbox_camera = self.bbox_cameras[self.idx_video-1]

            #{'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5}
            for i, bboxes in enumerate(bbox_camera):
                for b in bboxes:
                    if self.bbox_idx == b[0]:
                        camidx = i
                        bbox_coord = b[1]
                        break

            og_img_frame = self.og_imgs_frames[self.idx_video-1][camidx]
            label = self.bbox_labels[self.idx_video-1][self.bbox_idx]
            img_single_obj = og_img_frame[bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]

            all_expl = self.img_frames_attention_nobbx[self.idx_video-1]

            for i in range(len(all_expl)):
                expl = all_expl[i]
                expl = expl[camidx][bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]

                ax_img = self.fig_obj.add_subplot(self.single_object_spec[i, 0])
                ax_attn = self.fig_obj.add_subplot(self.single_object_spec[i, 1])

                ax_img.imshow(img_single_obj)
                ax_img.set_title(f"{self.class_names[label].capitalize()}")
                ax_attn.imshow(expl)
                ax_attn.set_title(f"{self.expl_types[i]}")

                ax_img.axis('off')
                ax_attn.axis("off")
            
            self.fig_obj.tight_layout(pad=0)
            self.obj_canvas.draw()
        
        self.after_obj_id = self.after(1, self.show_att_maps_object)

    def generate_video_frame(self):

        self.update_data()

        # Avoid selecting all layers and all cameras. Only the last layer will be visualized
        if self.show_all_layers.get() and self.selected_camera.get() == -1:
            self.selected_layer.set(self.Attention.layers - 1)

        # Extract the selected bounding box indexes from the menu
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        if self.selected_expl_type.get() in ["Grad-CAM", "Gradient Rollout"]:
            self.update_data()

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
            img = self.overlay_attention_on_image(img, attn, intensity=200)   

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cam_imgs.append(img)

        return cam_imgs, att_nobbx, og_imgs, bbox_cameras, self.labels
    
    def save_video(self):
        if hasattr(self, "img_frames"):
            data = {'img_frames': self.img_frames, 'img_frames_attention_nobbx': self.img_frames_attention_nobbx, 'og_imgs_frames': self.og_imgs_frames, 'bbox_cameras': self.bbox_cameras, 'bbox_labels': self.bbox_labels}

            file_path = fd.asksaveasfilename(defaultextension=".pkl", filetypes=[("All Files", "*.*")])

            print(f"Saving video in {file_path}...\n")
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            self.show_message(f"Video saved in {file_path}")
        else:
            self.show_message("You should first generate a video.")

    def load_video(self):
        video_datatypes = (
            ('Pickle', '*.pkl'),
        )
        video_pickle = fd.askopenfilename(
            title='Load video data',
            initialdir='/workspace/',
            filetypes=video_datatypes)

        print(f"Loading video from {video_pickle}...\n")
        with open(video_pickle, 'rb') as f:
            data = pickle.load(f)

        self.img_frames = data["img_frames"]
        self.img_frames_attention_nobbx = data["img_frames_attention_nobbx"]
        self.og_imgs_frames = data["og_imgs_frames"]
        self.bbox_cameras = data["bbox_cameras"]
        self.bbox_labels = data["bbox_labels"]

        self.show_message(f"Video loaded from {video_pickle}.\n")

        

        

    
