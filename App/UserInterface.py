import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import numpy as np
import cv2

from mmcv.parallel import DataContainer as DC
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from App.File import load_from_config, load_model
from App.Utils import show_message, show_model_info, red_text, black_text, \
                    select_data_idx, random_data_idx, update_thr, capture, \
                    single_bbox_select, update_scores, add_separator



class UserInterface(tk.Tk):

    def __init__(self):
        '''
        Tkinter initialization with model loading option.
        '''
        super().__init__()

        # Tkinter-related settings
        style = ttk.Style(self)
        style.theme_use("alt")
        self.title('Attention Visualization')
        self.geometry('1500x1500')
        self.protocol("WM_DELETE_WINDOW", lambda: (self.quit(), self.destroy())) # Terminate debug session after closing window
        self.canvas, self.fig, self.spec = None, None, None

        # Model and dataloader objects
        self.model, self.dataloader = None, None

        # Main Tkinter menu in which all other cascade menus are added
        self.menubar = tk.Menu(self)

        # Cascade menus for loading model and selecting the GPU
        self.config(menu=self.menubar)
        file_opt, gpu_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.gpu_id = tk.IntVar()
        self.gpu_id.set(0)
        file_opt.add_command(label="Load model", command=lambda:load_model(self))
        file_opt.add_command(label="Load from config file", command=lambda:load_from_config(self))
        file_opt.add_separator()
        file_opt.add_cascade(label="Gpu", menu=gpu_opt)
        message = "You need to reload the model to apply GPU change."
        for i in range(torch.cuda.device_count()):
            gpu_opt.add_radiobutton(label=f"GPU {i}", variable=self.gpu_id, value=i, command=lambda: show_message(self, message))

        self.menubar.add_cascade(label="File", menu=file_opt)

        # Speeding up the testing
        load_from_config(self)

    def start_app(self):
        '''
        It starts the UI after loading the model. Variables are initialized.
        '''

        # Booleans used for avoiding reloading same data so that the UI is speed up
        self.old_data_idx, self.old_thr, self.old_bbox_idx, self.old_expl_type, self.new_model = \
            None, None, None, None, None

        # Suffix used for saving screenshot of same model with different numbering
        self.file_suffix = 0

        # Tkinter frame for visualizing model and GPU info
        frame = tk.Frame(self)
        frame.pack(fill=tk.Y)
        self.info_text = tk.StringVar()
        self.info_label = tk.Label(frame, textvariable=self.info_text, anchor=tk.CENTER)
        self.info_label.bind("<Button-1>", lambda event, k=self:show_model_info(self))
        self.info_label.bind("<Enter>", lambda event, k=self:red_text(self))
        self.info_label.bind("<Leave>", lambda event, k=self:black_text(self))
        self.info_label.pack(side=tk.TOP)

        # Cascade menu for Data index
        dataidx_opt = tk.Menu(self.menubar)
        dataidx_opt.add_command(label="Select data index", command=lambda: select_data_idx(self))
        dataidx_opt.add_command(label="Select random data", command=lambda: random_data_idx(self))

        # Cascade menus for Prediction threshold and Discard ratio
        thr_opt, dr_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.selected_threshold, self.selected_discard_ratio = tk.DoubleVar(), tk.DoubleVar()
        self.selected_threshold.set(0.5)
        self.selected_discard_ratio.set(0.5)
        values = np.arange(0.0, 1, 0.1).round(1)
        for i in values:
            thr_opt.add_radiobutton(label=i, variable=self.selected_threshold, command=lambda: update_thr(self))
            dr_opt.add_radiobutton(label=i, variable=self.selected_discard_ratio)

        # Cascade menu for Camera
        camera_opt = tk.Menu(self.menubar)
        self.cameras = {'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5, 'All': 6}
        self.cam_idx = [2, 0, 1, 5, 3, 4]
        self.selected_camera = tk.IntVar()
        self.selected_camera.set(0)
        for value, key in enumerate(self.cameras):
            camera_opt.add_radiobutton(label=key, variable=self.selected_camera, value=value)

        # Cascade menus for Explainable options
        attn_opt, attn_rollout, grad_cam, grad_rollout = tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.expl_options = ["Attention Rollout", "Grad-CAM", "Gradient Rollout"]

        # Attention Rollout
        attn_opt.add_cascade(label=self.expl_options[0], menu=attn_rollout)
        self.head_types = ["mean", "min", "max"]
        self.selected_head_fusion = tk.StringVar()
        self.selected_head_fusion.set(self.head_types[2])
        self.raw_attn = tk.BooleanVar()
        self.raw_attn.set(True)
        for i in range(len(self.head_types)):
            attn_rollout.add_radiobutton(label=self.head_types[i].capitalize(), variable=self.selected_head_fusion, value=self.head_types[i])
        attn_rollout.add_radiobutton(label="All", variable=self.selected_head_fusion, value="all")
        attn_rollout.add_checkbutton(label="Raw attention", variable=self.raw_attn, onvalue=1, offvalue=0)

        # Grad-CAM
        attn_opt.add_cascade(label=self.expl_options[1], menu=grad_cam)
        self.grad_cam_types = ["default"]
        self.selected_gradcam_type = tk.StringVar()
        self.selected_gradcam_type.set(self.grad_cam_types[0])
        for i in range(len(self.grad_cam_types)):
            grad_cam.add_radiobutton(label=self.grad_cam_types[i].capitalize(), variable=self.selected_gradcam_type, value=self.grad_cam_types[i])

        # Gradient Rollout
        attn_opt.add_cascade(label=self.expl_options[2], menu=grad_rollout)
        self.grad_roll_types = ["default"]
        self.selected_gradroll_type = tk.StringVar()
        self.selected_gradroll_type.set(self.grad_roll_types[0])
        for i in range(len(self.grad_roll_types)):
            grad_rollout.add_radiobutton(label=self.grad_roll_types[i].capitalize(), variable=self.selected_gradroll_type, value=self.grad_roll_types[i])

        # Attention layer
        attn_layer = tk.Menu(self.menubar)
        attn_opt.add_cascade(label="Layer", menu=attn_layer)
        self.selected_layer = tk.IntVar()
        self.selected_layer.set(5)
        for i in range(len(self.model.module.pts_bbox_head.transformer.decoder.layers)):
            attn_layer.add_radiobutton(label=i, variable=self.selected_layer)
        attn_layer.add_radiobutton(label="All", variable=self.selected_layer, value=6)

        attn_opt.add_separator()

        # Explainable mechanism selection
        expl_opt = tk.Menu(self.menubar)
        attn_opt.add_cascade(label="Explainability mechanism", menu=expl_opt)
        self.selected_expl_type = tk.StringVar()
        self.selected_expl_type.set(self.expl_options[0])
        self.old_expl_type = self.expl_options[0]
        for i in range(len(self.expl_options)):
            expl_opt.add_radiobutton(label=self.expl_options[i], variable=self.selected_expl_type, value=self.expl_options[i])

        # Cascade menus for Bounding boxes
        self.bbox_opt = tk.Menu(self.menubar)
        self.single_bbox = tk.BooleanVar()
        self.single_bbox.set(True)
        self.bbox_opt.add_checkbutton(label="Single bounding box", onvalue=1, offvalue=0, variable=self.single_bbox)
        self.bbox_opt.add_separator()

        # Cascade menus for Additional options
        add_opt = tk.Menu(self.menubar)
        self.GT_bool, self.BB_bool, self.points_bool, self.show_scale, self.attn_contr, self.attn_norm, self.overlay, self.show_labels, self.capture_bool, self.bbox_2d = \
            tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.BB_bool.set(True)
        self.show_scale.set(True)
        self.show_labels.set(True)
        self.attn_norm.set(True)
        self.attn_contr.set(True)
        self.capture_bool.set(False)
        add_opt.add_checkbutton(label="Show GT Bounding Boxes", onvalue=1, offvalue=0, variable=self.GT_bool)
        add_opt.add_checkbutton(label="Show all Bounding Boxes", onvalue=1, offvalue=0, variable=self.BB_bool)
        add_opt.add_checkbutton(label="Show attention scale", onvalue=1, offvalue=0, variable=self.show_scale)
        add_opt.add_checkbutton(label="Show attention camera contributions", onvalue=1, offvalue=0, variable=self.attn_contr)
        add_opt.add_checkbutton(label="Normalize attention", onvalue=1, offvalue=0, variable=self.attn_norm)
        add_opt.add_checkbutton(label="Overlay attention on image", onvalue=1, offvalue=0, variable=self.overlay)
        add_opt.add_checkbutton(label="Show predicted labels", onvalue=1, offvalue=0, variable=self.show_labels)
        add_opt.add_checkbutton(label="Capture output", onvalue=1, offvalue=0, variable=self.capture_bool)
        add_opt.add_checkbutton(label="2D bounding boxes", onvalue=1, offvalue=0, variable=self.bbox_2d)

        # Adding all cascade menus ro the main menubar menu
        add_separator(self)
        self.menubar.add_cascade(label="Data", menu=dataidx_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Prediction threshold", menu=thr_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Discard ratio", menu=dr_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Camera", menu=camera_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Attention", menu=attn_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Bounding boxes", menu=self.bbox_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Options", menu=add_opt)
        add_separator(self, "|")
        self.menubar.add_command(label="Visualize", command=self.visualize)

    def show_attn_maps(self, grid_clm=1, ):
        '''
        Shows the attention map for explainability.
        '''
        # If attention contribution option is selected, the scores are updated
        if self.attn_contr.get():
            update_scores(self)

        # If all cameras are selected, generate their attentions and append them to a list
        attn_list = []
        for i in range(6):
            if self.selected_camera.get() == 6:
                attn = self.Attention.generate_explainability(self.selected_expl_type.get(), self.selected_layer.get(), self.bbox_idx, self.nms_idxs, self.cam_idx[i], self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get())
            elif self.selected_layer.get() == 6:
                attn = self.Attention.generate_explainability(self.selected_expl_type.get(), i, self.bbox_idx, self.nms_idxs, self.selected_camera.get(), self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get())
            else:
                attn = self.Attention.generate_explainability(self.selected_expl_type.get(), self.selected_layer.get(), self.bbox_idx, self.nms_idxs, self.selected_camera.get(), self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get())
                attn_list.append(attn)
                break
            attn_list.append(attn)     

        # If we want to visualize all layers or all cameras:
        if self.selected_layer.get() == 6 or self.selected_camera.get() == 6:
            # Select the center of the grid to plot the attentions and add 2x2 subgrid
            layer_grid = self.spec[1, grid_clm].subgridspec(2, 3) 
            fontsize = 8
        else:
            fontsize = 12
        
        attn_max = torch.max(torch.cat(attn_list)).cpu().numpy()

        for i in range(len(attn_list)):
            if len(attn_list) > 1:
                ax_attn = self.fig.add_subplot(layer_grid[i > 2, i if i < 3 else i - 3])
            else:
                ax_attn = self.fig.add_subplot(self.spec[1, grid_clm])

            attn = attn_list[i].view(29, 50).cpu().numpy()
            ax_attn.axis('off')

            #Attention normalization if option is selected
            if self.attn_norm.get():
                attn /= attn_max
                attmap = ax_attn.imshow(attn, vmin=0, vmax=1)
            else:
                attmap = ax_attn.imshow(attn)

            # Visualize attention bar scale if option is selected
            if self.show_scale.get():  
                im_ratio = attn.shape[1]/attn.shape[0]
                self.fig.colorbar(attmap, ax=ax_attn, orientation='horizontal', extend='both', fraction=0.047*im_ratio)

            # Set title accordinly
            if self.selected_layer.get() == 6:
                title = f'{list(self.cameras.keys())[self.selected_camera.get()]}, layer {i}, {self.selected_head_fusion.get()}'
            elif self.selected_camera.get() == 6:
                title = f'{list(self.cameras.keys())[self.cam_idx[i]]}, layer {self.selected_layer.get()}, {self.selected_head_fusion.get()}'
                if self.attn_contr.get():
                    title += f', {self.scores_perc[self.cam_idx[i]]}%'
            else:
                title = f'{list(self.cameras.keys())[self.selected_camera.get()]}, layer {self.Attention.layer}, {self.selected_head_fusion.get()}, {self.scores_perc[self.selected_camera.get()]}%'

            ax_attn.set_title(title, fontsize=fontsize)

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

        # Extract the 6 camera images from the data and remove the padded pixels
        imgs = self.data["img"][0]._data[0].numpy()[0]
        imgs = imgs.transpose(0, 2, 3, 1)[:, :900, :, :]
        self.imgs = imgs.astype(np.uint8)

        # Extract image metas which contain, for example, the lidar to camera projection matrices
        self.img_metas = self.data["img_metas"][0]._data[0][0]

        # Update the Bounding box menu with the predicted labels
        self.bboxes = []
        self.bbox_opt.delete(3, 'end')
        for i in range(len(self.thr_idxs.nonzero())):
            view_bbox = tk.BooleanVar()
            view_bbox.set(False)
            self.bboxes.append(view_bbox)
            self.bbox_opt.add_checkbutton(label=f"{self.class_names[self.labels[i].item()].capitalize()} ({i})", onvalue=1, offvalue=0, variable=self.bboxes[i], command=lambda idx=i: single_bbox_select(self, idx))
        
        # Default bbox for first visualization
        if self.bboxes:
            self.bboxes[0].set(True)

    def visualize(self):
        '''
        Visualizes predicted bounding boxes on all the cameras and shows
        the attention map in the middle of the plot.
        '''

        # Create figure with a 3x3 grid if not existent
        if self.fig is None:
            self.fig = plt.figure(figsize=(80, 60), layout="constrained")
            self.spec = self.fig.add_gridspec(3, 3)
        else: 
            self.fig.clear()

        # Data is updated only when data idx, prediction threshold or the model is changed
        if self.old_data_idx != self.data_idx or self.old_thr != self.selected_threshold.get() or self.old_expl_type != self.selected_expl_type.get() or self.new_model:
            self.update_data()
            if self.new_model:
                self.new_model = False
            if self.old_thr != self.selected_threshold.get():
                self.old_thr = self.selected_threshold.get()
            if self.old_data_idx != self.data_idx:
                self.old_data_idx = self.data_idx
            if self.old_expl_type != self.selected_expl_type.get():
                self.old_expl_type = self.selected_expl_type.get()
            
        # Extract the selected bounding box indexes from the menu
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]
        
        # Update the attention layer
        if self.selected_layer.get() != 6:
            self.Attention.layer = self.selected_layer.get()

        # Extract Ground Truth bboxes if wanted
        if self.GT_bool.get():
            self.gt_bbox = self.dataloader.dataset.get_ann_info(self.data_idx)['gt_bboxes_3d']
        else:
            self.gt_bbox = None

        # Show attention map 
        if self.selected_expl_type.get() == "Attention Rollout" and self.selected_head_fusion.get() == "all":
            for k in range(len(self.head_types)):
                self.selected_head_fusion.set(self.head_types[k])
                self.show_attn_maps(grid_clm=k)
            self.selected_head_fusion.set("all")
        else:
            self.show_attn_maps()

        # Generate images with bboxes on it
        self.imgs_bbox = []
        for camidx in range(len(self.imgs)):
            img = draw_lidar_bbox3d_on_img(
                    self.pred_bboxes,
                    self.imgs[camidx],
                    self.img_metas['lidar2img'][camidx],
                    self.img_metas,
                    color=(0, 255, 0),
                    with_label=self.show_labels.get(),
                    all_bbx=self.BB_bool.get(),
                    bbx_idx=self.bbox_idx,
                    mode_2d=self.bbox_2d.get())  

            if self.gt_bbox:
                img = draw_lidar_bbox3d_on_img(
                        self.gt_bbox,
                        img,
                        self.img_metas['lidar2img'][camidx],
                        self.img_metas,
                        color=(255, 0, 0),
                        mode_2d=self.bbox_2d.get())

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs_bbox.append(img)

        # Visualize them on the figure subplots
        for i in range(len(self.imgs)):
            if i < 3:
                ax = self.fig.add_subplot(self.spec[0, i])
            else:
                ax = self.fig.add_subplot(self.spec[2,i-3])

            ax.imshow(self.imgs_bbox[self.cam_idx[i]])
            ax.axis('off')

            if self.attn_contr.get():
                if self.selected_layer.get() == 6:
                    ax.set_title(f'{list(self.cameras.keys())[self.cam_idx[i]]}')
                else:
                    ax.set_title(f'{list(self.cameras.keys())[self.cam_idx[i]]}, {self.scores_perc[self.cam_idx[i]]}%')
            else:
                ax.set_title(f'{list(self.cameras.keys())[self.cam_idx[i]]}')

        # Create canvas with the figure embedded in it, and update it after each visualization
        if self.canvas is None:
            self.canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.canvas.get_tk_widget().pack()

        self.canvas.draw()

        # take a screenshot if the option is selected
        if self.capture_bool.get():
            capture(self)
