from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.transforms as mtransforms
import matplotlib


from modules.BaseApp import BaseApp, tk, np, cv2, plt, mmcv, torch, DC, pickle, draw_lidar_bbox3d_on_img
import os
from mmcv.cnn import xavier_init

class App(BaseApp):
    ''' Application User Interface. '''

    def __init__(self):
        ''' Tkinter initialization with model loading option. '''

        super().__init__()

        # Loading model and dataset from config.toml
        self.load_model(from_config=True)

    def visualize(self):
        ''' Visualizes predicted bounding boxes on all the cameras and shows
            the attention map in the middle of the plot. '''
        
        if self.canvas is None or self.video_gen_bool:
            # Create canvas with the figure embedded in it, and update it after each visualization
            if self.video_gen_bool:
                # Remove video canvas
                self.canvas.pack_forget()
                self.scale.pack_forget()
                self.info_label_video.pack_forget()
                for key in ['<space>', '<Right>', '<Left>', '<Up>', '<Down>']:
                    self.unbind(key)
                self.menubar.add_command(label="Show video", command=self.show_video)
                self.add_separator("|")
            self.fig = plt.figure()
            self.fig.set_facecolor(self.bg_color)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.fig.clear()

        # Extract some options from menu
        sancheck_layers = [layer.get() for layer in self.selected_sancheck_layer]
        self.selected_layers = [index for index, value in enumerate(sancheck_layers) if value == 1]

        # This uses the Configs class: if one element is changed, the function update_data() is invoked.
        self.data_configs.configs = [self.data_idx, self.selected_threshold.get(), self.ObjectDetector.model_name, self.selected_pert_step.get(), self.selected_pert_type.get(), self.selected_layers]

        if self.video_gen_bool:
            self.video_gen_bool = False

        # Extract the selected objects from menu
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        # Set up the canvas grid depending if a single object is selected or not
        if self.single_bbox.get():
            self.spec = self.fig.add_gridspec(3, 3, wspace=0, hspace=0)
        else:
            self.spec = self.fig.add_gridspec(2, 3, wspace=0, hspace=0)

        if not self.no_object:
            if self.selected_expl_type.get() in ["Grad-CAM", "Gradient Rollout"]:
                print("Calculating gradients...")
                self.update_data(gradients=True, initialize_bboxes=False)

            # This uses the Configs class: if one element is changed, the function generate_explainability() is invoked.
            self.expl_configs.configs = [self.selected_expl_type.get(), self.selected_head_fusion.get(), self.handle_residual.get(), self.apply_rule.get(), self.outputs]  
            
            # Generates the xai maps by using different user-defined parameters. 
            self.ExplainableModel.select_explainability(
                self.nms_idxs, self.bbox_idx,
                self.selected_discard_threshold.get(),
                self.selected_map_quality.get(),
                remove_pad=True,
                layer_fusion_method=self.selected_layer_fusion_type.get())

            if self.single_bbox.get():
                # Extract camera with highest attention
                cam_obj = self.get_object_camera()
                if cam_obj == -1:
                    self.show_message("Please change the selected options")
                    return
                self.selected_camera = cam_obj
                self.color_dict = None

                if self.selected_expl_type.get() == "Self Attention":
                    self.show_xai_self_attention()
        
        # Generate images list with bboxes on it
        print("Generating camera images...")
        self.cam_imgs, self.saliency_maps_objects = [], []
        with_labels = True
        for camidx in range(len(self.imgs)):

            if self.draw_bboxes.get() or self.single_bbox.get():
                img, bbox_coords = draw_lidar_bbox3d_on_img(
                        self.pred_bboxes,
                        self.imgs[camidx],
                        self.img_metas['lidar2img'][camidx],
                        color=(0, 255, 0),
                        color_dict=self.color_dict,
                        with_bbox_id=with_labels,
                        all_bbx=True,
                        bbx_idx=self.bbox_idx,
                        mode_2d=self.bbox_2d.get(),
                        labels=self.labels)
            else:
                img = self.imgs[camidx]

            if self.selected_expl_type.get() != "Self Attention" and self.single_bbox.get() and camidx == self.selected_camera:
                og_img = self.imgs[camidx].astype(np.uint8)
                for layer in range(len(self.ExplainableModel.xai_layer_maps)):
                    xai_map = self.ExplainableModel.xai_layer_maps[layer][camidx]
                    saliency_map = self.generate_saliency_map(og_img, xai_map)      
                    saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                    self.saliency_maps_objects.append(saliency_map)
                xai_map = self.ExplainableModel.xai_maps[camidx]
                if self.gen_segmentation.get():
                    xai_map = xai_map.numpy() * 255
                    xai_map = xai_map.astype(np.uint8)
                    _, xai_map = cv2.threshold(xai_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    xai_map[xai_map == 255] = 1
                    xai_map = torch.from_numpy(xai_map)
                saliency_map = self.generate_saliency_map(og_img, xai_map)      
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
                
            if self.selected_expl_type.get() != "Self Attention" and self.overlay_bool.get() and not self.no_object:
                xai_map = self.ExplainableModel.xai_maps[camidx]
                saliency_map = self.generate_saliency_map(img, xai_map)        
                saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
                self.cam_imgs.append(saliency_map)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.cam_imgs.append(img)

        # Visualize the generated images list on the figure subplots
        pert_ax = 1
        print("Plotting...")
        for i in range(len(self.imgs)):
            if i < 3:
                ax = self.fig.add_subplot(self.spec[0, i])
            else:
                if not self.single_bbox.get() or self.selected_expl_type.get() == "Self Attention":
                    ax = self.fig.add_subplot(self.spec[1, i-3])
                else:
                    ax = self.fig.add_subplot(self.spec[2, i-3])

            ax.imshow(self.cam_imgs[self.cam_idx[i]])
            ax.axis('off')
            if i == pert_ax:
                self.pert_ax = ax

            if self.selected_expl_type.get() != "Self Attention" and self.single_bbox.get():
                score = self.ExplainableModel.scores[self.cam_idx[i]]
                ax.axhline(y=0, color=self.bg_color, linewidth=10)
                ax.axhline(y=0, color='green', linewidth=10, xmax=score/100)

        if self.single_bbox.get() and self.selected_expl_type.get() != "Self Attention":
            self.show_xai_cross_attention()

        self.fig.tight_layout(pad=0)
        self.canvas.draw()
        
        print("Done.\n")
        torch.cuda.empty_cache() 

    def show_xai_self_attention(self):
        ''' Visualized self-attention scores of a selected object 
            with all other objects present in the scene. '''
        if self.tk.call("ttk::style", "theme", "use") == "azure-dark":
            text_color = "white"
        else:
            text_color = "black"

        num_layers = len(self.ExplainableModel.self_xai_maps)
        k = min(3, len(self.labels))
        group_width = 0.15 * k
        bar_width = 0.1
        ax = self.fig.add_subplot(self.spec[2, :])
        group_centers = torch.arange(num_layers) * group_width
        max_height = 0

        # First loop to find the maximum height across all layers
        for i in range(num_layers):
            query_self_attn = self.ExplainableModel.self_xai_maps[i]
            query_self_attn = query_self_attn[self.thr_idxs]
            topk_values, topk_indices = torch.topk(query_self_attn, k)
            max_height = max(max_height, (topk_values / topk_values.sum() * 100).max().item())

        max_height += 2
        cmap = plt.cm.get_cmap('Reds') 
        self.color_dict = []

        for i in range(num_layers):
            query_self_attn = self.ExplainableModel.self_xai_maps[i]
            query_self_attn = query_self_attn[self.thr_idxs]

            topk_values, topk_indices = torch.topk(query_self_attn, k)

            bar_x = group_centers[i] - (group_width - bar_width) / 2 + torch.arange(k) * bar_width
            bars = ax.bar(bar_x, topk_values / topk_values.sum() * 100, bar_width) 

            norm = plt.Normalize(vmin=0, vmax=k-1)
            sorted_indices = torch.argsort(topk_values)

            for j, bar in enumerate(bars):
                if topk_values[j] == topk_values.min():  
                    bar.set_color('grey')  
                else:
                    bar_color = cmap(norm(sorted_indices[j]))  
                    bar.set_color(bar_color)
            
            if i == num_layers - 1:
                for index in topk_indices:
                    self.color_dict.append(index)

            group_center = (bar_x[0] + bar_x[-1]) / 2
            ax.text(group_center, -5, f"Layer {i}", ha='center', va='bottom', color=text_color, fontsize = 10)

            # Set edge color for bbox_idx[0] if it is in the top-k indices
            if self.bbox_idx[0] in topk_indices:
                idx = (topk_indices == self.bbox_idx[0]).nonzero()[0]
                bars[idx].set_edgecolor("black")
                bars[idx].set_linewidth(1.5)

            # Add value labels to the bars
            min_font_size, max_font_size = 8, 12
            fontsize = max(min_font_size, max_font_size - len(bars) // 10)
            for j, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(topk_indices[j].item()),
                        ha='center', va='bottom', color=text_color, fontsize=fontsize)

            min_font_size = 6  # smallest font size to use
            max_font_size = 16  # largest font size to use
            fontsize = 10
            
            for j, bar in enumerate(bars):
                bar_width = bar.get_width()
                object_class = self.ObjectDetector.class_names[self.labels[topk_indices[j].item()]].replace("_", " ").upper()
                text_length = len(object_class)
                fontsize = max(min_font_size, min(max_font_size, int(bar_width * 10 / text_length)))

                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, object_class, ha='center', va='center', color='white', fontsize=fontsize)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set(xticks=[], yticks=[], facecolor='none')

    def show_xai_cross_attention(self):
        ''' Shows the saliency map for a selected object. '''

        if self.selected_expl_type.get() != "Gradient Rollout":
            # Select the center of the grid to plot the attentions and add 2x2 subgrid
            layer_grid = self.spec[1, 1].subgridspec(2, 3)
        else:
            layer_grid = self.spec[1, 1].subgridspec(1, 1)

        if len(self.selected_layers) == 0:
            for b in self.bbox_coords:
                if self.bbox_idx[0] == b[0]:
                    self.extr_bbox_coord = b[1]
                    break
        
        # Saves saliency maps in a folder if the option is activated in the settings
        if self.capture_object.get():
            class_name = self.ObjectDetector.class_names[self.labels[self.bbox_idx[0]].item()]
            folder_path = f"screenshots/maps/{self.ObjectDetector.model_name}/{self.data_idx}_{self.selected_expl_type.get().replace(' ', '_')}_{self.selected_layer_fusion_type.get()}"
            if len(self.selected_layers) > 0:
                folder_path += "_"
                for layer in self.selected_layers:
                    folder_path += f"{layer}"
            folder_path = os.path.join(folder_path, class_name)
            file_name = 0
            file_path = os.path.join(folder_path, str(file_name))
            while os.path.exists(file_path):
                file_name += 1
                file_name_new = file_name
                file_path = os.path.join(folder_path, str(file_name_new))
            folder_path = file_path
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
    
        for i in range(len(self.saliency_maps_objects)):

            att_nobbx_obj = self.saliency_maps_objects[i]
            att_nobbx_obj = att_nobbx_obj[self.extr_bbox_coord[1].clip(min=0):self.extr_bbox_coord[3], self.extr_bbox_coord[0].clip(min=0):self.extr_bbox_coord[2]]

            if i != len(self.saliency_maps_objects) - 1:
                ax_obj_layer = self.fig.add_subplot(layer_grid[i > 2, i if i < 3 else i - 3])
                ax_obj_layer.imshow(att_nobbx_obj, vmin=0, vmax=1)
                ax_obj_layer.axis('off')
            
            elif self.gen_segmentation.get():
                ax_obj_seg = self.fig.add_subplot(self.spec[1, 2])
                ax_obj_seg.imshow(att_nobbx_obj, vmin=0, vmax=1)
                ax_obj_seg.axis('off')             

            if self.capture_object.get():
                fig_save, ax_save = plt.subplots()
                ax_save.imshow(att_nobbx_obj, vmin=0, vmax=1)
                ax_save.axis('off')  # Turn off axis

                fig_name = f"layer_{i}.png"
                if i == len(self.saliency_maps_objects) - 1:
                    fig_name = "full.png"
                file_path = os.path.join(folder_path, str(fig_name))

                file_path += ".png"
                fig_save.savefig(file_path, transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close(fig_save)

        if self.capture_object.get():
            print(f"Saliency maps saved in {folder_path}.")

        self.fig.tight_layout()

    def show_lidar(self):
        ''' Shows LiDAR point clouds of the scene. '''
        file_name = f"{self.data_idx}_{self.ObjectDetector.model_name}"
        self.ObjectDetector.dataset.show_mod(self.outputs, index=self.data_idx, out_dir="LiDAR/", show_gt=self.GT_bool.get(), show=True, snapshot=False, file_name=file_name, pipeline=None, score_thr=self.selected_threshold.get())

    def show_video(self):
        ''' Shows the loaded video. '''
        if self.canvas and not self.video_gen_bool or not self.canvas:
            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()
            end_idx = self.menubar.index('end')
            self.menubar.delete(end_idx-1, end_idx)

            self.bind('<space>', self.pause_resume)
            self.bind('<Right>', self.update_index)
            self.bind('<Left>', self.update_index)
            self.bind('<Up>', self.update_index)
            self.bind('<Down>', self.update_index)

            self.canvas = tk.Canvas(self)
            self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.canvas_frame = self.canvas.create_image(0, 0, image=None, anchor='nw', tags="img_tag")
            self.canvas.update()
            self.video_gen_bool = True

        self.update_object_filter()
        self.info_label_video.pack_forget()
        self.info_label_video.pack(side=tk.TOP)
        self.paused = False
        self.old_w, self.old_h = None, None
        self.layer_idx = self.layers_video - 1
        self.flag = False
        self.delay = 20  # Initial delay
        
        self.show_sequence()
    