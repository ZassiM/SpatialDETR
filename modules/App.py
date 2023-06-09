from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from modules.BaseApp import BaseApp, tk, np, cv2, plt, mmcv, torch, DC, overlay_attention_on_image


class App(BaseApp):
    '''
    Application User Interface
    '''
    def __init__(self):
        '''
        Tkinter initialization with model loading option.
        '''
        super().__init__()

        # Speeding up the testing
        self.load_model(from_config=True)

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
            self.fig.set_facecolor(self.bg_color)
            self.spec = self.fig.add_gridspec(3, 3)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.video_gen_bool = False

        self.fig.clear()

        self.data_configs.configs = [self.data_idx, self.selected_threshold.get(), self.ObjectDetector.model_name]

        # Extract the selected bounding box indexes from the menu
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]
        
        if self.selected_expl_type.get() == "All":
            for i in range(len(self.expl_options)):
                self.selected_expl_type.set(self.expl_options[i])
                self.update_explainability()
                self.show_attention_maps(grid_column=i)      

        else:
            self.update_explainability()
            self.show_attention_maps()

        # Generate images list with bboxes on it
        print("Generating camera images...")
        self.cam_imgs = []
        for camidx in range(len(self.imgs)):
            img, _ = draw_lidar_bbox3d_on_img(
                    self.pred_bboxes,
                    self.imgs[camidx],
                    self.img_metas['lidar2img'][camidx],
                    color=(0, 255, 0),
                    with_bbox_id=self.show_labels.get(),
                    all_bbx=self.BB_bool.get(),
                    bbx_idx=self.bbox_idx,
                    mode_2d=self.bbox_2d.get())
            
            # Extract Ground Truth bboxes if wanted
            if self.GT_bool.get():
                self.gt_bbox = self.ObjectDetector.dataset.get_ann_info(self.data_idx)['gt_bboxes_3d']
                img, _ = draw_lidar_bbox3d_on_img(
                        self.gt_bbox,
                        img,
                        self.img_metas['lidar2img'][camidx],
                        color=(255, 0, 0),
                        mode_2d=self.bbox_2d.get())
                
            if self.overlay_bool.get() and ((self.selected_camera.get() != -1 and camidx == self.selected_camera.get()) or (self.selected_camera.get() == -1)):
                if self.show_all_layers.get():
                    attn = self.attn_list[self.selected_layer.get()][self.selected_camera.get()]
                elif self.selected_camera.get() == -1:
                    attn = self.attn_list[self.selected_layer.get()][camidx]
                else:
                    attn = self.attn_list[self.selected_layer.get()][self.selected_camera.get()]

                img = overlay_attention_on_image(img, attn)            

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.cam_imgs.append(img)

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

        # Take a screenshot if the option is selected
        if self.capture_bool.get():
            self.capture()
        
        del self
        torch.cuda.empty_cache()
        print("Done.\n")

    def update_explainability(self):
        # Avoid selecting all layers and all cameras. Only the last layer will be visualized
        if (self.show_all_layers.get() and self.selected_camera.get() == -1) or self.selected_expl_type.get() == "Gradient Rollout":
            self.show_all_layers.set(False)
            if self.selected_expl_type.get() == "Gradient Rollout":
                self.selected_layer.set(0)

        if self.selected_expl_type.get() in ["Grad-CAM", "Gradient Rollout"]:
            self.update_data(initialize_bboxes=False)
                
        self.expl_configs.configs = [self.selected_expl_type.get(), self.bbox_idx, self.nms_idxs, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get()]   
        self.attn_list = self.expl_configs.attn_list

    def show_attention_maps(self, grid_column=1):
        '''
        Shows the attention map for explainability.
        '''
        # If we want to visualize all layers or all cameras:
        if self.show_all_layers.get() or self.selected_camera.get() == -1:
            # Select the center of the grid to plot the attentions and add 2x2 subgrid
            layer_grid = self.spec[1, grid_column].subgridspec(2, 3)
            fontsize = 6
        else:
            fontsize = 12
            
        for i in range(len(self.attn_list[0])):
            if self.show_all_layers.get() or self.selected_camera.get() == -1:
                ax_attn = self.fig.add_subplot(layer_grid[i > 2, i if i < 3 else i - 3])
            else:
                ax_attn = self.fig.add_subplot(self.spec[1, grid_column])
            
            if self.show_all_layers.get():
                attn = self.attn_list[i][self.selected_camera.get()]
            elif self.selected_camera.get() == -1:
                attn = self.attn_list[self.selected_layer.get()][self.cam_idx[i]]
            else:
                attn = self.attn_list[self.selected_layer.get()][self.selected_camera.get()]

            ax_attn.imshow(attn, vmin=0, vmax=1)      

            # Set title accordinly
            if self.show_all_layers.get() or self.selected_camera.get() != -1:
                title = f'{list(self.cameras.keys())[self.selected_camera.get()]}'
                if self.selected_expl_type.get() != "Gradient Rollout":
                    title += f' | layer {i} '
            else:
                title = f'{list(self.cameras.keys())[self.cam_idx[i]]}'
                if self.selected_expl_type.get() != "Gradient Rollout":
                    title += f' | layer {self.selected_layer.get()}'

            # If doing Attention Rollout, visualize head fusion type
            if self.selected_expl_type.get() == "Attention Rollout":
                title += f' | head {self.selected_head_fusion.get()}'

            # Show attention camera contributon if all cameras are selected
            if self.attn_contr.get() and self.selected_camera.get() == -1:
                score = self.update_scores(self.cam_idx[i])
                title += f'| {score}%'
                ax_attn.axhline(y=0, color='black', linewidth=10)
                ax_attn.axhline(y=0, color='green', linewidth=10, xmax=score/100)

            if self.tk.call("ttk::style", "theme", "use") == "azure-dark":
                title_color = "white"
            else:
                title_color = "black"
            ax_attn.set_title(title, fontsize=fontsize, color=title_color, pad=0)
            ax_attn.axis('off')   
            self.fig.tight_layout()
   
            if not self.show_all_layers.get() and self.selected_camera.get() != -1:
                break

    def show_lidar(self):
        # self.attn_list[layer] = 6x900x1600
        '''
        For each attn camera, i have one attention value for each pixel. I have a matrix 6x1450x3 of depth values with respect to the 
        ref coordinate for each pixel. I can generate a point cloud for the attention maps in which I use xyz from the depth value for each
        pixel in each camera, and color it depending on the attention value
        '''
        #self.outputs, index=self.data_idx, out_dir="points/", show_gt=False, show=True, pipeline=None, score_thr = self.selected_threshold.get()))
        o3d_vis_id = self.after(0, self.ObjectDetector.dataset.show_mod, self.outputs, self.data_idx, "points/", False, True, None, 0.5)
        #self.after_cancel(o3d_vis_id)
        debug=0

    def show_video(self):
        if self.canvas and not self.video_gen_bool or not self.canvas:
            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            self.menubar.add_command(label="Pause/Resume", command=self.pause_resume)
            self.canvas = tk.Canvas(self)
            self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.canvas_frame = self.canvas.create_image(0, 0, anchor='nw', image=None)
            self.video_gen_bool = True

        if self.single_object_window is None and not self.video_only.get():
            self.single_object_window = tk.Toplevel(self)
            self.single_object_window.title("Object Explainability Visualizer")
            self.fig_obj = plt.figure()
            self.single_object_spec = self.fig_obj.add_gridspec(3, 2)
            self.obj_canvas = FigureCanvasTkAgg(self.fig_obj, master=self.single_object_window)
            self.obj_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.idx_video = 0
        self.paused = False

        if not hasattr(self, "img_frames"):
            if not self.video_only.get():
                self.generate_video()
            else:
                self.generate_video_only()
        
        self.old_bbox_idx = None
        self.old_layer = self.selected_layer.get()
        self.show_sequence()

    def generate_video(self):
        self.select_all_bboxes.set(True)
        self.img_frames, self.img_frames_attention_nobbx, self.og_imgs_frames, self.bbox_coords, self.bbox_cameras, self.bbox_labels, self.all_expl = [], [], [], [], [], [], []

        print("\nGenerating image frames...\n")
        prog_bar = mmcv.ProgressBar(self.video_length)
        for i in range(self.data_idx, self.data_idx + self.video_length):
            self.data_idx = i
            img_frames, img_att_nobbx = [], []
            for expl in self.expl_options:
                self.selected_expl_type.set(expl)
                imgs_att, imgs_att_nobbx, imgs_og, bbox_camera, labels = self.generate_video_frame()
                img_frames.append(imgs_att)
                img_att_nobbx.append(imgs_att_nobbx)

            self.img_frames.append(img_frames)  # Image frame of all 6 cameras with attention maps and bboxes overlayed
            self.img_frames_attention_nobbx.append(img_att_nobbx)  # Attention maps without bboxes
            self.og_imgs_frames.append(imgs_og)  # Original camera images
            self.bbox_cameras.append(bbox_camera)  # Coordinates of objects
            self.bbox_labels.append(labels)  # Labels for each frame

            prog_bar.update()

        self.data_idx -= (self.video_length - 1)
        print("\nVideo generated.\n")

    def generate_video_only(self):
        self.select_all_bboxes.set(True)
        self.img_frames = []

        print("\nGenerating image frames...\n")
        prog_bar = mmcv.ProgressBar(self.video_length)
        for i in range(self.data_idx, self.data_idx + self.video_length):
            self.data_idx = i
            img_frames = []
            for expl in self.expl_options:
                self.selected_expl_type.set(expl)
                imgs_att = self.generate_video_frame()
                img_frames.append(imgs_att)

            self.img_frames.append(img_frames)  # Image frame of all 6 cameras with attention maps and bboxes overlayed

            prog_bar.update()

        self.data_idx -= (self.video_length - 1)
        print("\nVideo generated.\n")

    def show_sequence(self):
        if not self.paused:
            img_frame = self.img_frames[self.idx_video][0]
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
            if not self.video_only.get():
                labels = self.bbox_labels[self.idx_video-1]
                self.update_objects_list(labels)  # Update objects menu list
                self.single_bbox_select(True)  # Selects the first object
                self.show_att_maps_object()  # Visualize attention maps for the selected object
        else:
            self.paused = False
            if not self.video_only.get():
                self.after_cancel(self.after_obj_id)
            self.show_sequence()

    def show_att_maps_object(self):
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        if (len(self.bbox_idx) == 1 and self.bbox_idx != self.old_bbox_idx) or self.old_layer != self.selected_layer.get():
            self.old_bbox_idx = self.bbox_idx
            self.fig_obj.clear()
            self.bbox_idx = self.bbox_idx[0]
            bbox_camera = self.bbox_cameras[self.idx_video-1]
            self.old_layer = self.selected_layer.get()

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
                if self.expl_options[i] == "Gradient Rollout":
                    expl = all_expl[i][camidx][0]
                else:
                    expl = all_expl[i][camidx][self.selected_layer.get()]
                expl = expl[bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]

                ax_img = self.fig_obj.add_subplot(self.single_object_spec[i, 0])
                ax_attn = self.fig_obj.add_subplot(self.single_object_spec[i, 1])

                ax_img.imshow(img_single_obj)
                ax_img.set_title(f"{self.ObjectDetector.class_names[label].capitalize()}")
                ax_img.axis('off')

                ax_attn.imshow(expl)
                title = f"{self.expl_options[i]}"
                if self.expl_options[i] != "Gradient Rollout":
                    title += f", layer {self.selected_layer.get()}"
                ax_attn.set_title(title)
                ax_attn.axis("off")
            
            self.fig_obj.tight_layout(pad=0)
            self.obj_canvas.draw()
        
        self.after_obj_id = self.after(1, self.show_att_maps_object)

    def generate_video_frame(self):

        self.update_data()

        # Extract the selected bounding box indexes from the menu
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        self.attn_list = self.ExplainableModel.generate_explainability(self.selected_expl_type.get(), self.bbox_idx, self.nms_idxs, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get())

        # Generate images list with bboxes on it
        cam_imgs, og_imgs, bbox_cameras, att_nobbx = [], [], [], []  # Used for video generation

        for camidx in range(len(self.imgs)):
            if not self.video_only.get():
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
            
            if not self.video_only.get():
                bbox_cameras.append(bbox_camera)  
            
            if self.selected_expl_type.get() == "Gradient Rollout":
                attn = self.attn_list[0][camidx]
            else:
                attn = self.attn_list[self.selected_layer.get()][camidx]
            img = overlay_attention_on_image(img, attn, intensity=200)   
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cam_imgs.append(img)     

        hori = np.concatenate((cam_imgs[2], cam_imgs[0], cam_imgs[1]), axis=1)
        ver = np.concatenate((cam_imgs[5], cam_imgs[3], cam_imgs[4]), axis=1)
        img_frame = np.concatenate((hori, ver), axis=0)

        if not self.video_only.get():
            return img_frame, att_nobbx, og_imgs, bbox_cameras, self.labels
        else:
            return img_frame
