from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from modules.BaseApp import BaseApp, tk, np, cv2, plt, mmcv, torch, DC, overlay_attention_on_image
import shutil
import os


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

        self.fig.clear()

        self.data_configs.configs = [self.data_idx, self.selected_threshold.get(), self.ObjectDetector.model_name]

        if self.video_gen_bool:
            self.video_gen_bool = False

        # Extract the selected bounding box indexes from the menu
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]
        
        self.update_explainability()

        if self.single_bbox.get() and (self.show_all_layers.get() or self.selected_camera.get() != -1) or not self.show_attn.get():
            # Extract camera with highest attention
            cam_obj = self.get_camera_object()
            self.selected_camera.set(cam_obj)
    
        # Generate images list with bboxes on it
        print("Generating camera images...")
        self.cam_imgs, self.bbox_coords, self.att_nobbx_all = [], [], []
        for camidx in range(len(self.imgs)):

            if not self.show_attn.get():
                og_img = self.imgs[camidx].astype(np.uint8)
                att_nobbx_layers = []
                for layer in range(len(self.ExplainableModel.attn_list)):
                    attn = self.ExplainableModel.attn_list[layer][camidx]
                    attn_img = overlay_attention_on_image(og_img, attn)      
                    attn_img = cv2.cvtColor(attn_img, cv2.COLOR_BGR2RGB)
                    att_nobbx_layers.append(attn_img)
                self.att_nobbx_all.append(att_nobbx_layers)

            img, bbox_coords = draw_lidar_bbox3d_on_img(
                    self.pred_bboxes,
                    self.imgs[camidx],
                    self.img_metas['lidar2img'][camidx],
                    color=(0, 255, 0),
                    with_bbox_id=self.show_labels.get(),
                    all_bbx=self.BB_bool.get(),
                    bbx_idx=self.bbox_idx,
                    mode_2d=self.bbox_2d.get())

            if not self.show_attn.get():
                self.bbox_coords.append(bbox_coords)

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
                    attn = self.ExplainableModel.attn_list[self.selected_layer.get()][self.selected_camera.get()]
                elif self.selected_camera.get() == -1:
                    attn = self.ExplainableModel.attn_list[self.selected_layer.get()][camidx]
                else:
                    attn = self.ExplainableModel.attn_list[self.selected_layer.get()][self.selected_camera.get()]

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
            if not self.show_attn.get() and self.single_bbox.get():
                score = self.update_scores()[self.cam_idx[i]]
                ax.axhline(y=0, color='black', linewidth=10)
                ax.axhline(y=0, color='green', linewidth=10, xmax=score/100)
            ax.axis('off')

        self.show_attention_maps()

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
        self.expl_configs.configs = [self.selected_expl_type.get(), self.bbox_idx, self.nms_idxs, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.apply_rollout.get(), self.handle_residual.get(), self.apply_rule.get()]   

    def show_attention_maps(self, grid_column=1):
        '''
        Shows the attention map for explainability.
        '''

        # If we want to visualize all layers or all cameras:
        if self.show_all_layers.get() or self.selected_camera.get() == -1:
            # Select the center of the grid to plot the attentions and add 2x2 subgrid
            layer_grid = self.spec[1, grid_column].subgridspec(2, 3)
            fontsize = 7
        else:
            fontsize = 12

        if self.tk.call("ttk::style", "theme", "use") == "azure-dark":
            title_color = "white"
        else:
            title_color = "black"

        if not self.show_attn.get():
            # cam_obj = self.get_camera_object()
            # self.selected_camera.set(cam_obj)
            bbox_coords = self.bbox_coords[self.selected_camera.get()]
            for b in bbox_coords:
                if self.bbox_idx[0] == b[0]:
                    bbox_coord = b[1]
                    break

            if self.show_all_layers.get():
                layer_grid = self.spec[1, grid_column].subgridspec(2, 3)
                for i in range(len(self.att_nobbx_all[0])):
                    ax_obj_layer = self.fig.add_subplot(layer_grid[i > 2, i if i < 3 else i - 3])

                    att_nobbx_obj = self.att_nobbx_all[self.selected_camera.get()][i]
                    att_nobbx_obj = att_nobbx_obj[bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]
                    ax_obj_layer.imshow(att_nobbx_obj, vmin=0, vmax=1)   

                    ax_obj_layer.set_title(f"layer {i}", fontsize=fontsize, color=title_color, pad=0)
                    ax_obj_layer.axis('off')

            else:
                att_nobbx_obj = self.att_nobbx_all[self.selected_camera.get()][self.selected_layer.get()]
                att_nobbx_obj = att_nobbx_obj[bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]
                ax_obj = self.fig.add_subplot(self.spec[1, grid_column])
                ax_obj.imshow(att_nobbx_obj, vmin=0, vmax=1)   
                ax_obj.set_title(f"layer {self.selected_layer.get()}", fontsize=fontsize, color=title_color, pad=0)
                ax_obj.axis('off')

        else:
            for i in range(len(self.ExplainableModel.attn_list[0])):
                if self.show_all_layers.get() or self.selected_camera.get() == -1:
                    ax_attn = self.fig.add_subplot(layer_grid[i > 2, i if i < 3 else i - 3])
                else:
                    ax_attn = self.fig.add_subplot(self.spec[1, grid_column])
                
                if self.show_all_layers.get():
                    attn = self.ExplainableModel.attn_list[i][self.selected_camera.get()]
                elif self.selected_camera.get() == -1:
                    attn = self.ExplainableModel.attn_list[self.selected_layer.get()][self.cam_idx[i]]
                else:
                    attn = self.ExplainableModel.attn_list[self.selected_layer.get()][self.selected_camera.get()]

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

                # Show attention camera contributon if all cameras are selected and one object is selected
                if self.selected_camera.get() == -1 and self.single_bbox.get():
                    score = self.update_scores()[self.cam_idx[i]]
                    title += f' | {score}%'
                    ax_attn.axhline(y=0, color='black', linewidth=10)
                    ax_attn.axhline(y=0, color='green', linewidth=10, xmax=score/100)

                ax_attn.set_title(title, fontsize=fontsize, color=title_color, pad=0)
                ax_attn.axis('off')   

                if not self.show_all_layers.get() and self.selected_camera.get() != -1:
                    break
        self.fig.tight_layout()

    def show_lidar(self):
        self.ObjectDetector.dataset.show_mod(self.outputs, index=self.data_idx, out_dir="points/", show_gt=self.GT_bool.get(), show=True, snapshot=False, pipeline=None, score_thr=self.selected_threshold.get())

    def show_video(self):
        if not hasattr(self, "img_frames"):
            generated = self.generate_video(save_img=True)
        else:
            generated = True
        
        if generated:
            if self.canvas and not self.video_gen_bool or not self.canvas:
                if self.canvas:
                    self.canvas.get_tk_widget().pack_forget()
                self.menubar.add_command(label="Pause/Resume", command=self.pause_resume)
                self.canvas = tk.Canvas(self)
                self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                self.canvas_frame = self.canvas.create_image(0, 0, anchor='nw', image=None)
                self.video_gen_bool = True

            self.data_idx = self.start_video_idx
            self.idx_video = 0
            self.paused = False

            self.show_sequence()

    def generate_video(self, save_img=True):
        if self.video_length > ((len(self.ObjectDetector.dataset)-1) - self.data_idx):
            self.show_message(f"Video lenght should be between 2 and {len(self.ObjectDetector.dataset) - self.data_idx}") 
            return False

        if save_img:
            folder = "video_gen"
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)

        self.select_all_bboxes.set(True)
        self.img_frames, self.img_labels = [], []
        self.start_video_idx = self.data_idx

        print("\nGenerating image frames...\n")
        prog_bar = mmcv.ProgressBar(self.video_length)
        for i in range(self.data_idx, self.data_idx + self.video_length):
            self.data_idx = i
            imgs_att, labels = self.generate_video_frame(folder=folder, save_img=save_img)
            self.img_frames.append(imgs_att)  # Image frame of all 6 cameras with attention maps and bboxes overlayed
            self.img_labels.append(labels)
            prog_bar.update()

        self.data_idx = self.start_video_idx
        print("\nVideo generated.\n")
        return True

    def show_sequence(self):
        if not self.paused:
            self.idx_video += 1
            if self.idx_video >= self.video_length:
                self.idx_video = 0

            img_frame = self.img_frames[self.idx_video]
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            self.img_frame = ImageTk.PhotoImage(Image.fromarray((img_frame * 255).astype(np.uint8)).resize((w, h)))
            self.canvas.itemconfig(self.canvas_frame, image=self.img_frame)
            self.update_info_label(idx=self.data_idx + self.idx_video)

            self.after_seq_id = self.after(self.frame_rate, self.show_sequence)

    def pause_resume(self):
        if not self.paused:
            self.after_cancel(self.after_seq_id)
            self.paused = True
            labels = self.img_labels[self.idx_video]
            self.update_objects_list(labels=labels)
            #self.single_bbox_select(idx=5)
            self.data_idx = self.start_video_idx + self.idx_video

        else:
            self.paused = False
            self.show_sequence()

    def generate_video_frame(self, folder, save_img=True):

        self.update_data(initialize_bboxes=False)

        # Extract the selected bounding box indexes from the menu
        self.bbox_idx = list(range(len(self.labels)))

        self.ExplainableModel.generate_explainability(self.selected_expl_type.get(), self.bbox_idx, self.nms_idxs, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.apply_rollout.get(), self.handle_residual.get(), self.apply_rule.get())

        # Generate images list with bboxes on it
        cam_imgs = []  # Used for video generation

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
            

            if self.selected_expl_type.get() == "Gradient Rollout":
                attn = self.ExplainableModel.attn_list[0][camidx]
            else:
                attn = self.ExplainableModel.attn_list[self.selected_layer.get()][camidx]

            img = overlay_attention_on_image(img, attn)   
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cam_imgs.append(img)     

        hori = np.concatenate((cam_imgs[2], cam_imgs[0], cam_imgs[1]), axis=1)
        ver = np.concatenate((cam_imgs[5], cam_imgs[3], cam_imgs[4]), axis=1)
        img_frame = np.concatenate((hori, ver), axis=0)

        if save_img:
            img = (img_frame * 255).astype(np.uint8)
            img = Image.fromarray(img)
            file_name = f"{self.selected_expl_type.get()}_{self.data_idx}.jpeg"
            file_path = os.path.join(folder, file_name)
            img.save(file_path)

        return img_frame, self.labels
