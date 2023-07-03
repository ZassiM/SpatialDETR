from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
import matplotlib.transforms as mtransforms


from modules.BaseApp import BaseApp, tk, np, cv2, plt, mmcv, torch, DC, generate_saliency_map, pickle
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
                end_idx = self.menubar.index('end')
                for key in ['<space>', '<Right>', '<Left>', '<Up>', '<Down>']:
                    self.unbind(key)

                self.menubar.delete(end_idx-1, end_idx)
            self.fig = plt.figure()
            self.fig.set_facecolor(self.bg_color)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.fig.clear()

        # if self.selected_expl_type.get() == "Gradient Rollout":
        #         self.selected_layer.set(0)

        self.data_configs.configs = [self.data_idx, self.selected_threshold.get(), self.ObjectDetector.model_name]

        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        if not self.no_object:
            if self.selected_expl_type.get() in ["Grad-CAM", "Gradient Rollout"]:
                print("Calculating gradients...")
                self.update_data(initialize_bboxes=False)

            self.expl_configs.configs = [self.selected_expl_type.get(), self.selected_head_fusion.get(), self.handle_residual.get(), self.apply_rule.get(), self.data_idx]  
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
                self.overlay_bool.set(False)
                self.spec = self.fig.add_gridspec(3, 3, wspace=0, hspace=0)
            else:
                self.overlay_bool.set(True)
                self.spec = self.fig.add_gridspec(2, 3, wspace=0, hspace=0)

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
                if self.aggregate_layers.get():
                    xai_map = self.ExplainableModel.xai_maps.max(dim=0)[0][camidx]
                else:
                    xai_map = self.ExplainableModel.xai_maps[-1][camidx]
                saliency_map = generate_saliency_map(img, xai_map)        
                saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                self.cam_imgs.append(saliency_map)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.cam_imgs.append(img)

        if self.single_bbox.get():
            self.overlay_bool.set(False)
            self.show_explainability()
        else:
            self.overlay_bool.set(True)

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
                score = self.ExplainableModel.scores[self.cam_idx[i]]
                ax.axhline(y=0, color='black', linewidth=10)
                ax.axhline(y=0, color='green', linewidth=10, xmax=score/100)

            ax.axis('off')

        if self.single_bbox.get():
            self.show_explainability()

        self.fig.tight_layout(pad=0)
        self.canvas.draw()
        
        print("Done.\n")
        torch.cuda.empty_cache() 

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
            ax_obj_layer.axis('off')
            
            if self.capture_object.get():
                fig_save, ax_save = plt.subplots()
                ax_save.imshow(att_nobbx_obj, vmin=0, vmax=1)
                ax_save.axis('off')  # Turn off axis
                class_name = self.ObjectDetector.class_names[self.labels[self.bbox_idx[0]].item()]
                folder_path = f"Thesis/{self.data_idx}_{self.selected_expl_type.get()}_{class_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                fig_save.savefig(os.path.join(folder_path, f"layer_{i}.png"), transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close(fig_save)
    

        
        if self.show_self_attention.get() and len(self.labels) > 1:

            # title = "Self-attention"
            # if self.selected_expl_type.get() != "Gradient Rollout":
            #     title += f" | layer {self.selected_layer.get()}"

            cmap = plt.cm.get_cmap('OrRd')  

            ax = self.fig.add_subplot(self.spec[1, 2])
            # num_objects = self.thr_idxs_layers[-1].sum()
            # queries_id = [nms_idxs[:num_objects] for nms_idxs in self.nms_îdxs_layers]
            # queries_scores = [scores[:num_objects] for scores in self.bbox_scores_layers]
            # #selected_id = queries_id[self.selected_layer.get()][self.bbox_idx[0]]
            # selected_id = queries_id[-1][self.bbox_idx[0]]
            # positions = []
            # for ids, scores in zip(queries_id, queries_scores):
            #     if selected_id in ids:
            #         index = (ids == selected_id).nonzero().item()
            #         positions.append(scores[index])
            #     else:
            #         positions.append(0)  # Lowest possible score

            # ax.bar(range(1, len(queries_id) + 1), positions, color=color_values)

            # ax.set_xlabel('Layers')
            # ax.set_ylim([self.selected_threshold.get(), ax.get_ylim()[1]])  # set the minimum y limit to y_limit
            # ax.set(xticks=[], yticks=[], facecolor='none')
            # for spine in ax.spines.values():
            #     spine.set_visible(False)
            #ax.set_title(f'Position of object {self.ObjectDetector.class_names[self.labels[self.bbox_idx[0]]]} in each layer')

            query_self_attn = self.ExplainableModel.self_xai_maps
            query_self_attn = query_self_attn[0]
            query_self_attn = query_self_attn[self.thr_idxs]
            norm = plt.Normalize(vmin=min(query_self_attn), vmax=max(query_self_attn))  # Use positions min and max for normalization
            color_values = cmap(norm(query_self_attn))

            x = torch.arange(len(query_self_attn))
            bars = ax.bar(x, query_self_attn / query_self_attn.sum() * 100)

            # Setting colors and highlighting bar
            [b.set_color(c) for b, c in zip(bars, color_values)]
            bars[self.bbox_idx[0]].set_edgecolor(text_color)
            bars[self.bbox_idx[0]].set_linewidth(1)

            # Setting various parameters
            ax.set(xticks=[], yticks=[], facecolor='none')
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Setting text labels
            min_font_size, max_font_size = 6, 10
            fontsize = max(min_font_size, max_font_size - len(bars) // 10)
            for i, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(i),
                        ha='center', va='bottom', color=text_color, fontsize=fontsize)
            
            #ax.set_title(title, color=text_color, fontsize=fontsize-4, y=0.95)
            ax2 = self.fig.add_subplot(self.spec[1, 0])
            # query_self_attn = self.ExplainableModel.self_xai_maps[self.selected_layer.get()]
            # query_self_attn = query_self_attn[0]
            # query_self_attn = query_self_attn[self.thr_idxs]
            norm = plt.Normalize(vmin=min(query_self_attn), vmax=max(query_self_attn))  # Use positions min and max for normalization
            color_values = cmap(norm(query_self_attn))
            edge_color = "black" if text_color == "white" else "white"
            labels = np.arange(len(self.labels))
            explode = [0.1 if i == self.bbox_idx[0] else 0 for i in range(len(self.labels))]
            _, texts = ax2.pie(query_self_attn, labels=labels, wedgeprops={'linewidth': 1.0, 'edgecolor': edge_color}, explode=explode, colors=color_values)
            for i in range(len(texts)):
                texts[i].set_color(text_color)
            #ax2.set_title(title, color=text_color, fontsize=fontsize-4, y=0.95)

        self.fig.tight_layout()

    def show_lidar(self):
        self.ObjectDetector.dataset.show_mod(self.outputs, index=self.data_idx, out_dir="points/", show_gt=self.GT_bool.get(), show=True, snapshot=False, pipeline=None, score_thr=self.selected_threshold.get())

    def show_video(self):
        if not hasattr(self, "img_frames"):
            generated = self.generate_video()
        else:
            generated = True
        
        if generated:
            if self.canvas and not self.video_gen_bool or not self.canvas:
                if self.canvas:
                    self.canvas.get_tk_widget().pack_forget()
                self.menubar.add_command(label="Pause/Resume", command=self.pause_resume)
                self.bind('<space>', self.pause_resume)
                self.bind('<Right>', self.update_index)
                self.bind('<Left>', self.update_index)
                self.bind('<Up>', self.update_index)
                self.bind('<Down>', self.update_index)

                self.add_separator("|")
                self.canvas = tk.Canvas(self)
                self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                self.canvas_frame = self.canvas.create_image(0, 0, image=None, anchor='nw', tags="img_tag")
                self.canvas.update()
                self.video_gen_bool = True
                       
            if hasattr(self, "scale"):
                self.scale.configure(to=self.video_length.get())
                self.idx_video.set(max(0, self.data_idx - self.start_video_idx))
            else:
                self.idx_video = tk.IntVar()
                self.scale = tk.Scale(self.frame, from_=0, to=self.video_length.get(), variable=self.idx_video, command=self.update_index, showvalue=False, orient='horizontal')
            self.scale.pack(fill='x')

            self.paused = False
            self.old_w, self.old_h = None, None
            self.layer_idx = self.layers_video - 1
            self.flag = False
            self.delay = 20  # Initial delay
            self.show_sequence()
    
    def update_index(self, event=None):
        if self.paused:
            if not isinstance(event, str):
                if self.flag: 
                    return
                self.flag = True
                self.after(self.delay, lambda: setattr(self, 'flag', False))  # Reset flag after delay
                if event.keysym == 'Right':
                    self.idx_video.set(self.idx_video.get() + 1)
                elif event.keysym == 'Left':
                    self.idx_video.set(self.idx_video.get() - 1)
                elif event.keysym == 'Up':
                    self.layer_idx = max(0, self.layer_idx - 1)
                elif event.keysym == 'Down':
                    self.layer_idx = min(self.layers_video-1, self.layer_idx + 1)
                self.frame.focus_set()
            self.show_sequence(forced=True)
            if hasattr(self, "img_labels"):
                labels = self.img_labels[self.idx_video.get()-1]
                self.update_objects_list(labels=labels, single_select=True)


    def pause_resume(self, event=None):
        if not self.paused:
            self.after_cancel(self.after_seq_id)
            self.paused = True
            if hasattr(self, "img_labels"):
                labels = self.img_labels[self.idx_video.get()-1]
                self.update_objects_list(labels=labels, single_select=True)

        else:
            self.paused = False
            self.show_sequence()

    def show_sequence(self, forced=False):
        if not self.paused or forced:
            if self.idx_video.get() >= self.video_length.get():
                self.idx_video.set(0)

            img_frame = self.img_frames[self.layer_idx][self.idx_video.get()]
            self.w, self.h = self.canvas.winfo_width(), self.canvas.winfo_height()

            if (self.old_w, self.old_h) != (self.w, self.h):
                img_w, img_h = img_frame.width, img_frame.height
                canvas_ratio, img_ratio = self.w / self.h, img_w / img_h

                if img_ratio > canvas_ratio:
                    self.new_w, self.new_h = self.w, int(self.w / img_ratio)
                else:
                    self.new_w, self.new_h = int(self.h * img_ratio), self.h

                x = (self.w - self.new_w) // 2
                y = (self.h - self.new_h) // 2
                self.canvas.coords("img_tag", x, y)

                self.old_w, self.old_h = self.w, self.h

            self.img_frame = ImageTk.PhotoImage(img_frame.resize((self.new_w, self.new_h)))
            self.canvas.itemconfig(self.canvas_frame, image=self.img_frame)
            
            self.data_idx = self.start_video_idx + self.idx_video.get()
            self.update_info_label()

            if not forced:
                self.idx_video.set(self.idx_video.get() + 1)
                self.after_seq_id = self.after(self.video_delay.get(), self.show_sequence)

                
    def generate_video(self):
        if self.video_length.get() > ((len(self.ObjectDetector.dataset)-1) - self.data_idx):
            self.show_message(f"Video lenght should be between 2 and {len(self.ObjectDetector.dataset) - self.data_idx}") 
            return False

        self.video_folder = f"videos/video_{self.data_idx}_{self.video_length.get()}"
        if os.path.isdir(self.video_folder):
            shutil.rmtree(self.video_folder)
        os.makedirs(self.video_folder)

        self.select_all_bboxes.set(True)
        self.img_labels = []
        self.start_video_idx = self.data_idx
        self.video_gen_bool = True
        print(f"\nGenerating video frames inside \"{self.video_folder}\"...")
        prog_bar = mmcv.ProgressBar(self.video_length.get())
        for i in range(self.data_idx, self.data_idx + self.video_length.get()):
            self.data_idx = i
            labels = self.generate_video_frame()
            self.img_labels.append(labels)
            prog_bar.update()

        data = {"img_labels": self.img_labels}
        file_path = os.path.join(self.video_folder, "labels.pkl")

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        self.data_idx = self.start_video_idx

        self.img_frames = []
        layer_folders = [f for f in os.listdir(self.video_folder) if f.startswith('layer_') and os.path.isdir(os.path.join(self.video_folder, f))]
        layer_folders.sort(key=lambda x: int(x.split('_')[-1]))  # Sort the folders by the layer number

        for folder in layer_folders:
            folder_path = os.path.join(self.video_folder, folder)
            folder_images = os.listdir(folder_path)
            folder_images.sort()
            images = [Image.open(os.path.join(folder_path, img)) for img in folder_images]
            self.img_frames.append(images)
        
        self.layers_video = len(self.img_frames)
            
        print(f"\nVideo generated inside \"{self.video_folder}\" folder.\n")

        return True
    
    def generate_video_frame(self):

        self.update_data(initialize_bboxes=False)

        # Extract the selected bounding box indices from the menu
        self.bbox_idx = list(range(len(self.labels)))
        self.ExplainableModel.generate_explainability(self.selected_expl_type.get(), self.selected_head_fusion.get(), self.handle_residual.get(), self.apply_rule.get())

        self.ExplainableModel.select_explainability(self.nms_idxs, self.bbox_idx, self.selected_discard_threshold.get(), self.selected_map_quality.get())

        # Generate images list with bboxes on it
        cam_imgs = []  
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
            
            if not self.no_object:
                cam_layers = []
                if self.aggregate_layers.get():
                    xai_map = self.ExplainableModel.xai_maps.max(dim=0)[0][camidx]
                    saliency_map = generate_saliency_map(img, xai_map)        
                    saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                    cam_layers.append(saliency_map)
                else:
                    for layer in range(len(self.ExplainableModel.xai_maps)):
                        xai_map = self.ExplainableModel.xai_maps[layer][camidx]
                        saliency_map = generate_saliency_map(img, xai_map)  
                        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                        cam_layers.append(saliency_map)

                cam_imgs.append(cam_layers)     

        for layer in range(len(cam_imgs[0])):
            hori = np.concatenate((cam_imgs[2][layer], cam_imgs[0][layer], cam_imgs[1][layer]), axis=1)
            ver = np.concatenate((cam_imgs[5][layer], cam_imgs[3][layer], cam_imgs[4][layer]), axis=1)
            img = np.concatenate((hori, ver), axis=0)
            if not self.no_object:
                img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            name = self.selected_expl_type.get().replace(" ", "_")
            file_name = f"{name}_{self.data_idx}.jpeg"
            layer_folder = os.path.join(self.video_folder, f"layer_{layer}")
            if not os.path.exists(layer_folder):
                os.makedirs(layer_folder)
            file_path = os.path.join(layer_folder, file_name)
            img.save(file_path)

        return self.labels
