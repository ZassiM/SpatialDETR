from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
import matplotlib.transforms as mtransforms


from modules.BaseApp import BaseApp, tk, np, cv2, plt, mmcv, torch, DC, generate_saliency_map
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
                self.scale.pack_forget()
                self.scale.destroy()
                end_idx = self.menubar.index('end')
                self.menubar.delete(end_idx-1, end_idx)
                self.video_gen_bool = False
            self.fig = plt.figure()
            self.fig.set_facecolor(self.bg_color)
            # self.spec = self.fig.add_gridspec(3, 3)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.fig.clear()

        if self.selected_expl_type.get() == "Gradient Rollout":
                self.selected_layer.set(0)

        self.data_configs.configs = [self.data_idx, self.selected_threshold.get(), self.ObjectDetector.model_name]

        if self.old_layer != self.selected_layer.get():
            if self.single_bbox.get():
                self.select_layer(initialize_bboxes=False)
            else:
                self.select_layer()
            self.old_layer = self.selected_layer.get()

        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        
        if not self.no_object:
            self.update_explainability()
            
            # if self.selected_pert_step.get() > 0:
            #     self.update_data(pert_step=self.selected_pert_step.get())

            # Extract the selected bounding box indices from the menu
            #self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

            self.ExplainableModel.select_explainability(self.nms_idxs, self.bbox_idx, self.selected_discard_threshold.get(), self.selected_map_quality.get())

            if self.selected_pert_step.get() > 0:
                self.update_data(pert_step=self.selected_pert_step.get())

            if self.single_bbox.get():
                # Extract camera with highest attention
                cam_obj = self.get_camera_object()
                if cam_obj == -1:
                    self.show_message("Please check the selected options.")
                    return
                self.selected_camera = cam_obj

            # _, indices = torch.topk(xai_map.flatten(), k=int(0.2*(1600*900)))
            # indices = indices[xai_map.flatten()[indices] > 0.5]
            # indices = np.array(np.unravel_index(indices.numpy(), xai_map.shape)).T
            # cols, rows = indices[:, 0], indices[:, 1]
            # saliency_map[cols, rows] = [0,0,0]

        # Generate images list with bboxes on it
        print("Generating camera images...")
        self.cam_imgs, self.saliency_maps_objects = [], []
        with_labels = True
        for camidx in range(len(self.imgs)):

            img, bbox_coords = draw_lidar_bbox3d_on_img(
                    self.pred_bboxes,
                    self.imgs[camidx],
                    self.img_metas['lidar2img'][camidx],
                    color=(0, 255, 0),
                    with_bbox_id=with_labels,
                    all_bbx=True,
                    bbx_idx=self.bbox_idx,
                    mode_2d=self.bbox_2d.get())

            if self.single_bbox.get() and camidx == self.selected_camera:
                og_img = self.imgs[camidx].astype(np.uint8)
                for layer in range(len(self.ExplainableModel.xai_maps)):
                    xai_map = self.ExplainableModel.xai_maps[layer][camidx]
                    if self.gen_segmentation.get():
                        xai_map = xai_map.numpy() * 255
                        xai_map = xai_map.astype(np.uint8)
                        _, xai_map = cv2.threshold(xai_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        xai_map[xai_map == 255] = 1
                        xai_map = torch.from_numpy(xai_map)
                    saliency_map = generate_saliency_map(og_img, xai_map)      
                    saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                    self.saliency_maps_objects.append(saliency_map)

                self.bbox_coords = bbox_coords

            # Extract Ground Truth bboxes if wanted
            if self.GT_bool.get():
                self.gt_bbox = self.ObjectDetector.dataset.get_ann_info(self.data_idx)['gt_bboxes_3d']
                img, _ = draw_lidar_bbox3d_on_img(
                        self.gt_bbox,
                        img,
                        self.img_metas['lidar2img'][camidx],
                        color=(255, 0, 0),
                        mode_2d=self.bbox_2d.get())
                
            if self.overlay_bool.get() and not self.no_object:
                xai_map = self.ExplainableModel.xai_maps[self.selected_layer.get()][camidx]
                saliency_map = generate_saliency_map(img, xai_map)        
                saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                self.cam_imgs.append(saliency_map)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.cam_imgs.append(img)

        if not self.single_bbox.get():
            self.spec = self.fig.add_gridspec(2, 3, wspace=0, hspace=0)
        else:
            self.spec = self.fig.add_gridspec(3, 3, wspace=0, hspace=0)

        # Visualize the generated images list on the figure subplots
        print("Plotting...")
        for i in range(len(self.imgs)):
            if i < 3:
                ax = self.fig.add_subplot(self.spec[0, i])
            else:
                if not self.single_bbox.get():
                    ax = self.fig.add_subplot(self.spec[1, i-3])
                else:
                    ax = self.fig.add_subplot(self.spec[2, i-3])

            ax.imshow(self.cam_imgs[self.cam_idx[i]])

            if self.single_bbox.get():
                score = self.ExplainableModel.scores[self.selected_layer.get()][self.cam_idx[i]]
                ax.axhline(y=0, color='black', linewidth=10)
                ax.axhline(y=0, color='green', linewidth=10, xmax=score/100)

            ax.axis('off')

        if self.single_bbox.get():
            self.show_explainability()

        self.fig.tight_layout(pad=0)
        self.canvas.draw()
        
        print("Done.\n")
        torch.cuda.empty_cache()


    def update_explainability(self):
        # Avoid selecting all layers and all cameras. Only the last layer will be visualized
        if self.selected_expl_type.get() in ["Grad-CAM", "Gradient Rollout"]:
            print("Calculating gradients...")
            self.update_data(initialize_bboxes=False)

        self.expl_configs.configs = [self.selected_expl_type.get(), self.selected_head_fusion.get(), self.handle_residual.get(), self.apply_rule.get(), self.data_idx]   


    def show_explainability(self):
        '''
        Shows the saliency map for explainability.
        '''
        if self.selected_expl_type.get() != "Gradient Rollout":
            # Select the center of the grid to plot the attentions and add 2x2 subgrid
            layer_grid = self.spec[1, 1].subgridspec(2, 3)
            fontsize = 8
        else:
            layer_grid = self.spec[1, 1].subgridspec(1, 1)
            fontsize = 12

        if self.tk.call("ttk::style", "theme", "use") == "azure-dark":
            text_color = "white"
        else:
            text_color = "black"

        for b in self.bbox_coords:
            if self.bbox_idx[0] == b[0]:
                bbox_coord = b[1]
                break
        
        for i in range(len(self.saliency_maps_objects)):
            ax_obj_layer = self.fig.add_subplot(layer_grid[i > 2, i if i < 3 else i - 3])

            att_nobbx_obj = self.saliency_maps_objects[i]
            att_nobbx_obj = att_nobbx_obj[bbox_coord[1].clip(min=0):bbox_coord[3], bbox_coord[0].clip(min=0):bbox_coord[2]]
            ax_obj_layer.imshow(att_nobbx_obj, vmin=0, vmax=1)   
            
            title = ""
            if self.selected_expl_type.get() != "Gradient Rollout":
                title = f"layer {i}"
                if self.selected_expl_type.get() == "Raw Attention":
                    title += f" | head {self.selected_head_fusion.get()}"

            ax_obj_layer.axis('off')
            #ax_obj_layer.set_title(title, fontsize=fontsize, color=text_color, pad=0)
        
        if self.show_self_attention.get():
            # Query self-attention visualization
            query_self_attn = self.ExplainableModel.self_xai_maps[self.selected_layer.get()]
            query_self_attn = query_self_attn[0]
            query_self_attn = query_self_attn[self.thr_idxs]

            title = "Self-attention"
            if self.selected_expl_type.get() != "Gradient Rollout":
                title += f" | layer {self.selected_layer.get()}"

            ax = self.fig.add_subplot(self.spec[1, 2])
            percentage = query_self_attn / query_self_attn.sum() * 100
            x = torch.arange(len(query_self_attn))
            bars = ax.bar(x, percentage)
            bars[self.bbox_idx[0]].set_color('red')
            ax.set_facecolor('none')
            ax.set_xticks(x)
            ax.set_yticks([])
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            ax.set_xticklabels([])
            min_font_size = 6
            max_font_size = 12
            num_bars = len(bars)
            fontsize = max(min_font_size, max_font_size - num_bars // 10)
            for i, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(i), ha='center', va='bottom', color=text_color,
                        fontsize=fontsize)
            
            edge_color = "black" if text_color == "white" else "white"
            title_color = edge_color
            ax.set_title(title, fontsize=fontsize-1, color=title_color)

            ax2 = self.fig.add_subplot(self.spec[1, 0])

            cmap = plt.cm.get_cmap('OrRd')  # Choose the desired colormap

            # Normalize the values of query_self_attn between 0 and 1
            norm = plt.Normalize(vmin=query_self_attn.min(), vmax=query_self_attn.max())
            color_values = cmap(norm(query_self_attn))
            labels = np.arange(len(self.labels))
            explode = [0.1 if i == self.bbox_idx[0] else 0 for i in range(len(self.labels))]
            patches, texts = ax2.pie(query_self_attn, labels=labels, wedgeprops={'linewidth': 1.0, 'edgecolor': edge_color}, explode=explode, colors=color_values)
            for i in range(len(texts)):
                texts[i].set_fontweight('bold')  # make the text bold
                texts[i].set_color(patches[i].get_facecolor())
            ax2.set_title(title, color=title_color, fontsize=fontsize-1)

        self.fig.tight_layout()

    def show_lidar(self):
        self.ObjectDetector.dataset.show_mod(self.outputs[self.selected_layer.get()], index=self.data_idx, out_dir="points/", show_gt=self.GT_bool.get(), show=True, snapshot=False, pipeline=None, score_thr=self.selected_threshold.get())

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
                self.add_separator("|")
                self.canvas = tk.Canvas(self)
                self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                #self.canvas_frame = self.canvas.create_image(0, 0, anchor='nw', image=None)
                self.canvas_frame = self.canvas.create_image(0, 0, image=None, anchor='nw', tags="img_tag")
                self.canvas.update()
                self.video_gen_bool = True
    
            if hasattr(self, "scale"):
                self.scale.configure(to=self.video_length.get())
            else:
                self.scale = tk.Scale(self.frame, from_=0, to=self.video_length.get(), showvalue=False, orient='horizontal', command=self.update_index)
                self.scale.pack(fill='x')

            self.data_idx = self.start_video_idx
            self.idx_video = 0
            self.paused = False
            self.old_w, self.old_h = None, None
            self.show_sequence()
    
    def update_index(self, value):
        if self.paused:
            self.idx_video = int(value)
            self.show_sequence(forced=True)

    def generate_video(self, save_img=True):
        if self.video_length.get() > ((len(self.ObjectDetector.dataset)-1) - self.data_idx):
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

        print("\nGenerating video frames...\n")
        prog_bar = mmcv.ProgressBar(self.video_length.get())
        for i in range(self.data_idx, self.data_idx + self.video_length.get()):
            self.data_idx = i
            imgs_att, labels = self.generate_video_frame(folder=folder, save_img=save_img)
            self.img_frames.append(imgs_att)  # Image frame of all 6 cameras with attention maps and bboxes overlayed
            self.img_labels.append(labels)
            prog_bar.update()

        self.data_idx = self.start_video_idx
        print("\nVideo generated.\n")
        return True

    def show_sequence(self, forced=False):
        if not self.paused or forced:
            if not forced:
                self.idx_video += 1
            if self.idx_video >= self.video_length.get():
                self.idx_video = 0

            img_frame = self.img_frames[self.idx_video]
            self.w, self.h = self.canvas.winfo_width(), self.canvas.winfo_height()

            if self.old_w != self.w or self.old_h != self.h:
                canvas_ratio = self.w / self.h
                img_w, img_h = img_frame.shape[1], img_frame.shape[0]
                img_ratio = img_w / img_h

                if img_ratio > canvas_ratio:
                    self.new_w = self.w
                    self.new_h = int(self.new_w / img_ratio)
                else:
                    self.new_h = self.h
                    self.new_w = int(self.new_h * img_ratio)

                # Center image in canvas
                x = (self.w - self.new_w) // 2
                y = (self.h - self.new_h) // 2
                self.canvas.coords("img_tag", x, y)

                self.old_w = self.w
                self.old_h = self.h

            self.img_frame = ImageTk.PhotoImage(Image.fromarray((img_frame * 255).astype(np.uint8)).resize((self.new_w, self.new_h)))
            self.canvas.itemconfig(self.canvas_frame, image=self.img_frame)

            self.update_info_label(idx=self.data_idx + self.idx_video)
            self.scale.set(self.idx_video)

            self.after_seq_id = self.after(self.frame_rate.get(), self.show_sequence)

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

        # Extract the selected bounding box indices from the menu
        self.bbox_idx = list(range(len(self.labels)))
        self.ExplainableModel.generate_explainability(self.selected_expl_type.get(), self.selected_head_fusion.get(), self.handle_residual.get(), self.apply_rule.get())

        self.ExplainableModel.select_explainability(self.nms_idxs, self.bbox_idx, self.selected_discard_threshold.get(), self.selected_map_quality.get())

        # Generate images list with bboxes on it
        cam_imgs = []  # Used for video generation

        for camidx in range(len(self.imgs)):

            img, _ = draw_lidar_bbox3d_on_img(
                    self.pred_bboxes,
                    self.imgs[camidx],
                    self.img_metas['lidar2img'][camidx],
                    color=(0, 255, 0),
                    with_bbox_id=True,
                    all_bbx=True,
                    bbx_idx=self.bbox_idx,
                    mode_2d=self.bbox_2d.get())
            
            if self.selected_expl_type.get() == "Gradient Rollout":
                attn = self.ExplainableModel.xai_maps[0][camidx]
            else:
                attn = self.ExplainableModel.xai_maps[self.selected_layer.get()][camidx]

            img = generate_saliency_map(img, attn)   
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
