
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from modules.BaseApp import BaseApp, tk, np, cv2, plt, mmcv, torch, DC, overlay_attention_on_image

class VideoGen():
    def show_video(self, App):
        if self.App.canvas and not self.video_gen_bool or not self.App.canvas:
            if self.App.canvas:
                self.App.canvas.get_tk_widget().pack_forget()
            self.App.menubar.add_command(label="Pause/Resume", command=self.pause_resume)
            self.App.canvas = tk.Canvas(self)
            self.App.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.App.canvas_frame = self.App.canvas.create_image(0, 0, anchor='nw', image=None)
            self.video_gen_bool = True

        if self.single_object_window is None and not self.App.video_only.get():
            self.single_object_window = tk.Toplevel(self)
            self.single_object_window.title("Object Explainability Visualizer")
            self.fig_obj = plt.figure()
            self.single_object_spec = self.fig_obj.add_gridspec(3, 2)
            self.obj_canvas = FigureCanvasTkAgg(self.fig_obj, master=self.single_object_window)
            self.obj_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.idx_video = 0
        self.paused = False

        if not hasattr(self, "img_frames"):
            if not self.App.video_only.get():
                self.generate_video()
            else:
                self.generate_video_only()
        
        self.old_bbox_idx = None
        self.old_layer = self.App.selected_layer.get()
        self.show_sequence()

    def generate_video(self):
        self.App.select_all_bboxes.set(True)
        self.img_frames, self.img_frames_attention_nobbx, self.og_imgs_frames, self.bbox_coords, self.bbox_cameras, self.App.bbox_labels, self.all_expl = [], [], [], [], [], [], []

        print("\nGenerating image frames...\n")
        prog_bar = mmcv.ProgressBar(self.App.video_length)
        for i in range(self.App.data_idx, self.App.data_idx + self.App.video_length):
            self.App.data_idx = i
            img_frames, img_att_nobbx = [], []
            for expl in self.App.expl_options:
                self.selected_expl_type.set(expl)
                imgs_att, imgs_att_nobbx, imgs_og, bbox_camera, labels = self.generate_video_frame()
                img_frames.append(imgs_att)
                img_att_nobbx.append(imgs_att_nobbx)

            self.img_frames.append(img_frames)  # Image frame of all 6 cameras with attention maps and bboxes overlayed
            self.img_frames_attention_nobbx.append(img_att_nobbx)  # Attention maps without bboxes
            self.og_imgs_frames.append(imgs_og)  # Original camera images
            self.bbox_cameras.append(bbox_camera)  # Coordinates of objects
            self.App.bbox_labels.append(labels)  # Labels for each frame

            prog_bar.update()

        self.App.data_idx -= (self.App.video_length - 1)
        print("\nVideo generated.\n")

    def generate_video_only(self):
        self.App.select_all_bboxes.set(True)
        self.img_frames = []

        print("\nGenerating image frames...\n")
        prog_bar = mmcv.ProgressBar(self.App.video_length)
        for i in range(self.App.data_idx, self.App.data_idx + self.App.video_length):
            self.App.data_idx = i
            img_frames = []
            for expl in self.App.expl_options:
                self.selected_expl_type.set(expl)
                imgs_att = self.generate_video_frame()
                img_frames.append(imgs_att)

            self.img_frames.append(img_frames)  # Image frame of all 6 cameras with attention maps and bboxes overlayed

            prog_bar.update()

        self.App.data_idx -= (self.App.video_length - 1)
        print("\nVideo generated.\n")

    def show_sequence(self):
        if not self.paused:
            img_frame = self.img_frames[self.idx_video][0]
            w, h = self.App.canvas.winfo_width(), self.App.canvas.winfo_height()
            self.img_frame = ImageTk.PhotoImage(Image.fromarray((img_frame * 255).astype(np.uint8)).resize((w, h)))
            self.App.canvas.itemconfig(self.App.canvas_frame, image=self.img_frame)
            self.idx_video += 1
            self.update_info_label(idx=self.App.data_idx + self.idx_video)

            if self.idx_video >= self.App.video_length:
                self.idx_video = 0

            self.App.after_seq_id = self.App.after(self.frame_rate, self.show_sequence)

    def pause_resume(self):
        if not self.paused:
            self.paused = True
            self.App.after_cancel(self.App.after_seq_id)
            if not self.App.video_only.get():
                labels = self.App.bbox_labels[self.idx_video-1]
                self.update_objects_list(labels)  # Update objects menu list
                self.single_bbox_select(True)  # Selects the first object
                self.show_att_maps_object()  # Visualize attention maps for the selected object
        else:
            self.paused = False
            if not self.App.video_only.get():
                self.App.after_cancel(self.App.after_obj_id)
            self.show_sequence()

    def show_att_maps_object(self):
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        if (len(self.bbox_idx) == 1 and self.bbox_idx != self.old_bbox_idx) or self.old_layer != self.App.selected_layer.get():
            self.old_bbox_idx = self.bbox_idx
            self.fig_obj.clear()
            self.bbox_idx = self.bbox_idx[0]
            bbox_camera = self.bbox_cameras[self.idx_video-1]
            self.old_layer = self.App.selected_layer.get()

            #{'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5}
            for i, bboxes in enumerate(bbox_camera):
                for b in bboxes:
                    if self.bbox_idx == b[0]:
                        camidx = i
                        bbox_coord = b[1]
                        break

            og_img_frame = self.og_imgs_frames[self.idx_video-1][camidx]
            label = self.App.bbox_labels[self.idx_video-1][self.bbox_idx]
            img_single_obj = og_img_frame[bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]

            all_expl = self.img_frames_attention_nobbx[self.idx_video-1]

            for i in range(len(all_expl)):
                if self.App.expl_options[i] == "Gradient Rollout":
                    expl = all_expl[i][camidx][0]
                else:
                    expl = all_expl[i][camidx][self.App.selected_layer.get()]
                expl = expl[bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]

                ax_img = self.fig_obj.add_subplot(self.single_object_spec[i, 0])
                ax_attn = self.fig_obj.add_subplot(self.single_object_spec[i, 1])

                ax_img.imshow(img_single_obj)
                ax_img.set_title(f"{self.ObjectDetector.class_names[label].capitalize()}")
                ax_img.axis('off')

                ax_attn.imshow(expl)
                title = f"{self.App.expl_options[i]}"
                if self.App.expl_options[i] != "Gradient Rollout":
                    title += f", layer {self.App.selected_layer.get()}"
                ax_attn.set_title(title)
                ax_attn.axis("off")
            
            self.fig_obj.tight_layout(pad=0)
            self.obj_canvas.draw()
        
        self.App.after_obj_id = self.App.after(1, self.show_att_maps_object)

    def generate_video_frame(self):

        self.update_data()

        # Extract the selected bounding box indexes from the menu
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        self.attn_list = self.ExplainableModel.generate_explainability(self.selected_expl_type.get(), self.bbox_idx, self.nms_idxs, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get())

        # Generate images list with bboxes on it
        cam_imgs, og_imgs, bbox_cameras, att_nobbx = [], [], [], []  # Used for video generation

        for camidx in range(len(self.imgs)):
            if not self.App.video_only.get():
                og_img = cv2.cvtColor(self.imgs[camidx], cv2.COLOR_BGR2RGB)
                og_imgs.append(og_img)
                og_img = self.imgs[camidx].astype(np.uint8)
                att_nobbx_layers = []
                for layer in range(len(self.attn_list)):
                    attn = self.attn_list[layer][camidx]
                    attn_img = overlay_attention_on_image(og_img, attn)      
                    attn_img = cv2.cvtColor(attn_img, cv2.COLOR_BGR2RGB)
                    att_nobbx_layers.append(attn_img)
                att_nobbx.append(att_nobbx_layers)

            img, bbox_camera = draw_lidar_bbox3d_on_img(
                    self.pred_bboxes,
                    self.imgs[camidx],
                    self.img_metas['lidar2img'][camidx],
                    color=(0, 255, 0),
                    with_bbox_id=self.show_labels.get(),
                    all_bbx=self.BB_bool.get(),
                    bbx_idx=self.bbox_idx,
                    mode_2d=self.bbox_2d.get())
            
            if not self.App.video_only.get():
                bbox_cameras.append(bbox_camera)  
            
            if self.selected_expl_type.get() == "Gradient Rollout":
                attn = self.attn_list[0][camidx]
            else:
                attn = self.attn_list[self.App.selected_layer.get()][camidx]
            img = overlay_attention_on_image(img, attn, intensity=200)   
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cam_imgs.append(img)     

        hori = np.concatenate((cam_imgs[2], cam_imgs[0], cam_imgs[1]), axis=1)
        ver = np.concatenate((cam_imgs[5], cam_imgs[3], cam_imgs[4]), axis=1)
        img_frame = np.concatenate((hori, ver), axis=0)

        if not self.App.video_only.get():
            return img_frame, att_nobbx, og_imgs, bbox_cameras, self.labels
        else:
            return img_frame
