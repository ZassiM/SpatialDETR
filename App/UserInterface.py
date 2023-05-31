import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import numpy as np
import cv2
import mmcv

from mmcv.parallel import DataContainer as DC
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

from App.File import load_from_config, load_model
from App.Utils import show_message, show_model_info, red_text, black_text, \
                    select_data_idx, random_data_idx, update_thr, capture, \
                    single_bbox_select, update_scores, add_separator, \
                    initialize_bboxes, update_info_label, overlay_attention_on_image, \
                    change_theme

from PIL import Image, ImageTk
import sys
import os


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class App(tk.Tk):
    '''
    Application User Interface
    '''
    def __init__(self):
        '''
        Tkinter initialization with model loading option.
        '''
        super().__init__()

        # Tkinter-related settings
        self.tk.call("source", "theme/azure.tcl")
        self.tk.call("set_theme", "light")
        self.title('Explainable Transformer-based 3D Object Detector')
        self.geometry('1500x1500')
        self.canvas, self.video_canvas, self.fig, self.spec = None, None, None, None

        # Model and dataloader objects
        self.model, self.dataloader = None, None
        self.started_app = False
        self.gen_video_bool = False
        self.video_length = 5
        
        # Main Tkinter menu in which all other cascade menus are added
        self.menubar = tk.Menu(self)

        # Cascade menus for loading model and selecting the GPU
        self.config(menu=self.menubar)
        file_opt, gpu_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.gpu_id = tk.IntVar()
        self.gpu_id.set(0)
        file_opt.add_command(label=" Load model", command=lambda:load_model(self))
        file_opt.add_command(label=" Load from config file", command=lambda:load_from_config(self))
        file_opt.add_separator()
        file_opt.add_cascade(label=" Gpu", menu=gpu_opt)
        message = "You need to reload the model to apply GPU change."
        for i in range(torch.cuda.device_count()):
            gpu_opt.add_radiobutton(label=f"GPU {i}", variable=self.gpu_id, value=i, command=lambda: show_message(self, message))

        self.menubar.add_cascade(label=" File", menu=file_opt)

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
        self.info_label.bind("<Button-1>", lambda event, k=self: show_model_info(self))
        self.info_label.bind("<Enter>", lambda event, k=self: red_text(self))
        self.info_label.bind("<Leave>", lambda event, k=self: black_text(self))
        self.info_label.pack(side=tk.TOP)

        # Cascade menu for Data index
        dataidx_opt = tk.Menu(self.menubar)
        dataidx_opt.add_command(label=" Select data index", command=lambda: select_data_idx(self))
        dataidx_opt.add_command(label=" Select video length", command=lambda: select_data_idx(self, length=True))
        dataidx_opt.add_command(label=" Select random data", command=lambda: random_data_idx(self))

        # Cascade menus for Prediction threshold
        thr_opt = tk.Menu(self.menubar)
        self.selected_threshold = tk.DoubleVar()
        self.selected_threshold.set(0.5)
        values = np.arange(0.0, 1, 0.1).round(1)
        for i in values:
            thr_opt.add_radiobutton(label=i, variable=self.selected_threshold, command=lambda: update_thr(self))

        # Cascade menu for Camera
        camera_opt = tk.Menu(self.menubar)
        self.cameras = {'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5}
        self.cam_idx = [2, 0, 1, 5, 3, 4]  # Used for visualizing camera outputs properly
        self.selected_camera = tk.IntVar()
        for value, key in enumerate(self.cameras):
            camera_opt.add_radiobutton(label=key, variable=self.selected_camera, value=value, command=lambda k=self: update_info_label(k))
        camera_opt.add_radiobutton(label="All", variable=self.selected_camera, value=-1, command=lambda k=self: update_info_label(k))
        self.selected_camera.set(-1) # Default: visualize all cameras

        # Cascade menu for Attention layer
        layer_opt = tk.Menu(self.menubar)
        self.selected_layer = tk.IntVar()
        self.show_all_layers = tk.BooleanVar()
        for i in range(self.Attention.layers):
            layer_opt.add_radiobutton(label=i, variable=self.selected_layer, command=lambda k=self: update_info_label(k))
        layer_opt.add_checkbutton(label="All", onvalue=1, offvalue=0, variable=self.show_all_layers)
        self.selected_layer.set(self.Attention.layers - 1)

        # Cascade menus for Explainable options
        expl_opt = tk.Menu(self.menubar)
        attn_rollout, grad_cam, partial_lrp, grad_rollout = tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.expl_options = ["Attention Rollout", "Grad-CAM", "Gradient Rollout"]

        # Attention Rollout
        expl_opt.add_cascade(label=self.expl_options[0], menu=attn_rollout)
        self.head_types = ["mean", "min", "max"]
        self.selected_head_fusion = tk.StringVar()
        self.selected_head_fusion.set(self.head_types[2])
        self.raw_attn = tk.BooleanVar()
        self.raw_attn.set(True)
        dr_opt, hf_opt = tk.Menu(self.menubar), tk.Menu(self.menubar)
        self.selected_discard_ratio = tk.DoubleVar()
        self.selected_discard_ratio.set(0.5)
        values = np.arange(0.0, 1, 0.1).round(1)
        for i in values:
            dr_opt.add_radiobutton(label=i, variable=self.selected_discard_ratio)
        for i in range(len(self.head_types)):
            hf_opt.add_radiobutton(label=self.head_types[i].capitalize(), variable=self.selected_head_fusion, value=self.head_types[i], command=lambda k=self: update_info_label(k))
        hf_opt.add_radiobutton(label="All", variable=self.selected_head_fusion, value="all", command=lambda k=self: update_info_label(k))
        attn_rollout.add_cascade(label=" Head fusion", menu=hf_opt)
        attn_rollout.add_cascade(label=" Discard ratio", menu=dr_opt)
        attn_rollout.add_checkbutton(label=" Raw attention", variable=self.raw_attn, onvalue=1, offvalue=0)

        # Grad-CAM
        expl_opt.add_cascade(label=self.expl_options[1], menu=grad_cam)
        self.grad_cam_types = ["default"]
        self.selected_gradcam_type = tk.StringVar()
        self.selected_gradcam_type.set(self.grad_cam_types[0])
        for i in range(len(self.grad_cam_types)):
            grad_cam.add_radiobutton(label=self.grad_cam_types[i].capitalize(), variable=self.selected_gradcam_type, value=self.grad_cam_types[i])

        # Gradient Rollout
        expl_opt.add_cascade(label=self.expl_options[2], menu=grad_rollout)
        self.handle_residual, self.apply_rule = tk.BooleanVar(),tk.BooleanVar()
        self.handle_residual.set(True)
        self.apply_rule.set(True)
        grad_rollout.add_checkbutton(label=" Handle residual", variable=self.handle_residual, onvalue=1, offvalue=0)
        grad_rollout.add_checkbutton(label=" Apply rule 10", variable=self.apply_rule, onvalue=1, offvalue=0)

        expl_opt.add_separator()

        # Explainable mechanism selection
        expl_type_opt = tk.Menu(self.menubar)
        expl_opt.add_cascade(label="Mechanism", menu=expl_type_opt)
        self.selected_expl_type = tk.StringVar()
        self.selected_expl_type.set(self.expl_options[0])
        self.old_expl_type = self.expl_options[0]
        for i in range(len(self.expl_options)):
            expl_type_opt.add_radiobutton(label=self.expl_options[i], variable=self.selected_expl_type, value=self.expl_options[i], command=lambda: update_info_label(self))

        expl_opt.add_command(label= "Evaluate explainability", command=self.evaluate_expl)

        # Cascade menus for object selection
        self.bbox_opt = tk.Menu(self.menubar)
        self.single_bbox = tk.BooleanVar()
        self.select_all_bboxes = tk.BooleanVar()
        self.single_bbox.set(True)
        self.bbox_opt.add_checkbutton(label=" Single object", onvalue=1, offvalue=0, variable=self.single_bbox) 
        self.bbox_opt.add_checkbutton(label=" Select all", onvalue=1, offvalue=0, variable=self.select_all_bboxes, command=lambda: initialize_bboxes(self))
        self.bbox_opt.add_separator()

        # Cascade menus for Additional options
        add_opt = tk.Menu(self.menubar)
        self.GT_bool, self.BB_bool, self.points_bool, self.show_scale, self.attn_contr, self.overlay_bool, self.show_labels, self.capture_bool, self.bbox_2d, self.dark_theme = \
            tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.BB_bool.set(True)
        self.show_labels.set(True)
        self.overlay_bool.set(True)
        self.bbox_2d.set(True)
        add_opt.add_checkbutton(label=" Show GT Bounding Boxes", onvalue=1, offvalue=0, variable=self.GT_bool)
        add_opt.add_checkbutton(label=" Show all Bounding Boxes", onvalue=1, offvalue=0, variable=self.BB_bool)
        add_opt.add_checkbutton(label=" Show attention scale", onvalue=1, offvalue=0, variable=self.show_scale)
        add_opt.add_checkbutton(label=" Show attention camera contributions", onvalue=1, offvalue=0, variable=self.attn_contr)
        add_opt.add_checkbutton(label=" Overlay attention on image", onvalue=1, offvalue=0, variable=self.overlay_bool)
        add_opt.add_checkbutton(label=" Show predicted labels", onvalue=1, offvalue=0, variable=self.show_labels)
        add_opt.add_checkbutton(label=" Capture output", onvalue=1, offvalue=0, variable=self.capture_bool)
        add_opt.add_checkbutton(label=" 2D bounding boxes", onvalue=1, offvalue=0, variable=self.bbox_2d)
        add_opt.add_checkbutton(label=" Dark theme", onvalue=1, offvalue=0, variable=self.dark_theme, command=lambda: change_theme(self))

        # Adding all cascade menus ro the main menubar menu
        add_separator(self)
        self.menubar.add_cascade(label="Data", menu=dataidx_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Prediction threshold", menu=thr_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Camera", menu=camera_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Objects", menu=self.bbox_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Layer", menu=layer_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Explainability", menu=expl_opt)
        add_separator(self)
        self.menubar.add_cascade(label="Options", menu=add_opt)
        add_separator(self, "|")
        self.menubar.add_command(label="Visualize", command=self.visualize)
        add_separator(self, "|")
        self.menubar.add_command(label="Generate video", command=self.gen_video)

        # Create figure with a 3x3 grid
        self.fig = plt.figure()
        self.spec = self.fig.add_gridspec(3, 3)
        self.single_object_spec = self.fig.add_gridspec(3, 2)
        self.spec.update(wspace=0, hspace=0)
        # Create canvas with the figure embedded in it, and update it after each visualization
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def evaluate_expl(self):
        print(f"Evaluating {self.selected_expl_type.get()}...")
        discard_ratio = 0.1
        bbox_idx = [0]
        initial_idx = 200
        len_dataset = 20
        mask = torch.zeros(3)
        prog_bar = mmcv.ProgressBar(len_dataset)
        
        for i in range(initial_idx, initial_idx + len_dataset):
            self.fig.clear()

            data = self.dataloader.dataset[i]
            metas = [[data['img_metas'][0].data]]
            img = [data['img'][0].data.unsqueeze(0)]
            data['img_metas'][0] = DC(metas, cpu_only=True)
            data['img'][0] = DC(img)

            # Attention scores are extracted, together with gradients if grad-CAM is selected
            if self.selected_expl_type.get() not in ["Grad-CAM", "Gradient Rollout"]:
                outputs = self.Attention.extract_attentions(data)

            self.nms_idxs = self.model.module.pts_bbox_head.bbox_coder.get_indexes()
            
            attn = self.Attention.generate_explainability(self.selected_expl_type.get(), self.selected_layer.get(), bbox_idx, self.nms_idxs, self.selected_camera.get(), self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get())

            attn = attn.view(1, 1, 29, 50)
            attn = torch.nn.functional.interpolate(attn, scale_factor=32, mode='bilinear')
            attn = attn.view(attn.shape[2], attn.shape[3]).cpu()
            _, indices = torch.topk(attn.flatten(), int(1450*discard_ratio))
            indices = np.array(np.unravel_index(indices.numpy(), attn.shape)).T

            # Mask input image
            img = img[0][0][self.selected_camera.get()].permute(1, 2, 0)
            img_pert = img
            for idx in indices:
                img_pert[idx[0], idx[1]] = mask

            # data['img'][0] = DC(img_pert)
            # outputs_perturbed = self.Attention.extract_attentions(data)

            ax_og = self.fig.add_subplot(self.spec[0, 0])
            ax_pert = self.fig.add_subplot(self.spec[1, 0])
            ax_att = self.fig.add_subplot(self.spec[2, 0])

            mean = np.array(self.img_norm_cfg["mean"], dtype=np.float32)
            std = np.array(self.img_norm_cfg["std"], dtype=np.float32)

            img = mmcv.imdenormalize(img.numpy(), mean, std, to_bgr=False)
            img_pert = mmcv.imdenormalize(img_pert.numpy(), mean, std, to_bgr=False)

            ax_og.imshow(img.astype(np.uint8))
            ax_pert.imshow(img_pert.astype(np.uint8))
            ax_att.imshow(attn)
        
            self.fig.tight_layout()
            self.canvas.draw()

            prog_bar.update()
            
            

    def gen_video(self):
        self.fig.clear()
        
        if self.video_canvas is None:
            self.video_canvas = tk.Canvas(self)
            self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.menubar.add_command(label="Pause/Resume", command=self.pause_resume)
            self.menubar.add_command(label="Restart", command=self.restart)

        self.thsObj = self.video_canvas.create_image(0, 0, anchor='nw', image=None)

        self.gen_video_bool = True
        self.select_all_bboxes.set(True)
        self.img_frames, self.img_frames_overlayed, self.og_imgs_frames, self.bbox_coords, self.bbox_cameras, self.bbox_labels, self.all_expl = [], [], [], [], [], [], []

        self.paused = False

        print("\nGenerating image frames...\n")
        prog_bar = mmcv.ProgressBar(self.video_length)
        blockPrint()

        expl_options = ["Attention Rollout"]

        for i in range(self.data_idx, self.data_idx + self.video_length):
            self.data_idx = i
            expl_object = []

            for expl_type in expl_options:
                self.selected_expl_type.set(expl_type)
                imgs, imgs_og, bbox_camera, labels = self.visualize()
                expl_object.append(imgs)

            self.img_frames_overlayed.append(expl_object)

            hori = np.concatenate((imgs[2], imgs[0], imgs[1]), axis=1)
            ver = np.concatenate((imgs[5], imgs[3], imgs[4]), axis=1)
            full = np.concatenate((hori, ver), axis=0)

            self.img_frames.append(full)
            self.og_imgs_frames.append(imgs_og)
            self.bbox_cameras.append(bbox_camera)
            self.bbox_labels.append(labels)
            prog_bar.update()

        enablePrint()

        self.idx_video = 0
        self.data_idx -= (self.video_length - 1)
        
        self.after("idle", self.show_sequence)

    def restart(self):
        self.idx_video = 0
        self.paused = False
        self.show_sequence()

    def show_sequence(self):

        if not self.paused:
            update_info_label(self, idx=self.data_idx + self.idx_video)
            img_frame = self.img_frames[self.idx_video]

            w, h = self.video_canvas.winfo_width(), self.video_canvas.winfo_height()
            self.img_frame = ImageTk.PhotoImage(Image.fromarray((img_frame * 255).astype(np.uint8)).resize((w, h)))
            self.video_canvas.itemconfig(self.thsObj, image=self.img_frame)

            self.idx_video += 1

            if self.idx_video < self.video_length:
                self.after(1, self.show_sequence)
            else:
                print("\nEnd\n")
                self.gen_video_bool = False
            
        else:
            labels = self.bbox_labels[self.idx_video-1]
            self.bboxes = []
            self.bbox_opt.delete(3, 'end')
            for i in range(len(labels)):
                view_bbox = tk.BooleanVar()
                view_bbox.set(False)
                self.bboxes.append(view_bbox)
                self.bbox_opt.add_checkbutton(label=f" {self.class_names[labels[i].item()].capitalize()} ({i})", onvalue=1, offvalue=0, variable=self.bboxes[i], command=lambda idx=i: single_bbox_select(self, idx))
            
            self.after(1, self.show_att_maps_object)
               
    def show_att_maps_object(self):

        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        if len(self.bbox_idx) == 1:
            self.bbox_idx = self.bbox_idx[0]
            self.video_canvas.destroy()
            self.fig.clear()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            bbox_camera = self.bbox_cameras[self.idx_video-1]

            #{'Front': 0, 'Front-Right': 1, 'Front-Left': 2, 'Back': 3, 'Back-Left': 4, 'Back-Right': 5}
            for i, bboxes in enumerate(bbox_camera):
                for b in bboxes:
                    if self.bbox_idx == b[0]:
                        camidx = i
                        bbox_coord = b[1]
                        break

            og_img_frame = self.og_imgs_frames[self.idx_video-1][camidx]
            
            img_single_obj = og_img_frame[bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]

            all_expl = self.img_frames_overlayed[self.idx_video]
            for i in range(len(all_expl)):
                attn = all_expl[i][camidx][bbox_coord[1]:bbox_coord[3], bbox_coord[0]:bbox_coord[2]]

                ax_img = self.fig.add_subplot(self.single_object_spec[i, 0])
                ax_attn = self.fig.add_subplot(self.single_object_spec[i, 1])

                ax_img.imshow(img_single_obj)
                ax_attn.imshow(attn)
                ax_img.set_title(f"{self.class_names[self.bbox_idx].capitalize()} ({self.bbox_idx})")
                ax_attn.set_title(f"{self.expl_options[i]}")
                ax_img.axis('off')
                ax_attn.axis("off")
            
            self.fig.tight_layout(pad=0)
            self.canvas.draw()
        
        self.after(1, self.show_att_maps_object)
        
    def pause_resume(self):
        if not self.paused:
            print(f"\nPaused at idx {self.data_idx + self.idx_video - 1}.\n")
            self.paused = True
        else:
            self.paused = False
            if not self.video_canvas:
                self.canvas.destroy()
                self.video_canvas = tk.Canvas(self)
                self.video_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                self.thsObj = self.video_canvas.create_image(0, 0, anchor='nw', image=None)
            self.after(1, self.show_sequence)

    def show_attention_maps(self, grid_clm=1):
        '''
        Shows the attention map for explainability.
        '''
        print("Generating attention maps...")
        # List to which attention maps are appended
        self.attn_list = []

        if self.selected_expl_type.get() == "Gradient Rollout":
            self.update_data()
            self.show_all_layers.set(False)
        

        # Explainable attention maps generation
        for i in range(6):
            # All cameras option
            if self.selected_camera.get() == -1:
                attn = self.Attention.generate_explainability(self.selected_expl_type.get(), self.selected_layer.get(), self.bbox_idx, self.nms_idxs, i, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get())
            
            # All layers option
            elif self.show_all_layers.get():
                attn = self.Attention.generate_explainability(self.selected_expl_type.get(), i, self.bbox_idx, self.nms_idxs, self.selected_camera.get(), self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get())
            
            # Single camera and single layer option
            else:
                attn = self.Attention.generate_explainability(self.selected_expl_type.get(), self.selected_layer.get(), self.bbox_idx, self.nms_idxs, self.selected_camera.get(), self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get(), self.handle_residual.get(), self.apply_rule.get())

            attn = attn.view(1, 1, 29, 50)
            attn = torch.nn.functional.interpolate(attn, scale_factor=32, mode='bilinear')
            attn = attn.view(attn.shape[2], attn.shape[3]).cpu()
            attn = attn[:900, :]

            # step = pert_steps[1]
            # curr_num_boxes = int(step * attn.size(-2) * attn.size(-1))
            # _, indices = torch.topk(attn.flatten(), k=curr_num_boxes)
            # indices = np.array(np.unravel_index(indices.numpy(), attn.shape)).T
            # self.topk_idx.append(indices)

            self.attn_list.append(attn.numpy())

            if self.selected_camera.get() != -1 and not self.show_all_layers.get():
                break
        
        # Extract maximum score for normalization
        attn_max = np.max(np.concatenate(self.attn_list))

        # If we want to visualize all layers or all cameras:
        if self.show_all_layers.get() or self.selected_camera.get() == -1:
            # Select the center of the grid to plot the attentions and add 2x2 subgrid
            layer_grid = self.spec[1, grid_clm].subgridspec(2, 3)
            fontsize = 8
        else:
            fontsize = 12

        # View attention maps
        if not self.gen_video_bool:
            self.topk_idx = []
            pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            for i in range(len(self.attn_list)):
                if self.show_all_layers.get() or self.selected_camera.get() == -1:
                    ax_attn = self.fig.add_subplot(layer_grid[i > 2, i if i < 3 else i - 3])
                else:
                    ax_attn = self.fig.add_subplot(self.spec[1, grid_clm])
                
                if self.selected_camera.get() == -1:
                    attn = self.attn_list[self.cam_idx[i]]
                else:
                    attn = self.attn_list[i]
                
                ax_attn.axis('off')

                # Attention map normalization
                if self.selected_camera.get() == -1:
                    attn /= attn_max
                    attmap = ax_attn.imshow(attn, vmin=0, vmax=1)
                else:
                    attn -= attn.min()
                    attn /= attn.max()
                    attmap = ax_attn.imshow(attn)

                step = pert_steps[1]
                attn = torch.from_numpy(attn)
                curr_num_boxes = int(step * attn.size(-2) * attn.size(-1))
                _, indices = torch.topk(attn.flatten(), k=curr_num_boxes)
                indices = np.array(np.unravel_index(indices.numpy(), attn.shape)).T
                self.topk_idx.append(indices)
                attn = attn.numpy()

                # Visualize attention bar scale if option is selected
                if self.show_scale.get():  
                    im_ratio = attn.shape[1]/attn.shape[0]
                    self.fig.colorbar(attmap, ax=ax_attn, orientation='horizontal', extend='both', fraction=0.047*im_ratio)

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
                if self.attn_contr.get() and self.selected_camera.get() == -1 and self.single_bbox.get():
                    update_scores(self)
                    title += f', {self.scores_perc[self.cam_idx[i]]}%'

                ax_attn.set_title(title, fontsize=fontsize)

    def update_data(self):
        '''
        Predict bboxes and extracts attentions.
        '''
        print("Detecting bounding boxes...")
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
        mean = np.array(self.img_norm_cfg["mean"], dtype=np.float32)
        std = np.array(self.img_norm_cfg["std"], dtype=np.float32)

        for i in range(len(imgs)):
            imgs[i] = mmcv.imdenormalize(imgs[i], mean, std, to_bgr=False)
        self.imgs = imgs.astype(np.uint8)

        # Extract image metas which contain, for example, the lidar to camera projection matrices
        self.img_metas = self.data["img_metas"][0]._data[0][0]

        # Update the Bounding box menu with the predicted labels
        if self.old_data_idx != self.data_idx or self.old_thr != self.selected_threshold.get() or self.new_model:
            self.bboxes = []
            self.bbox_opt.delete(3, 'end')
            for i in range(len(self.labels)):
                view_bbox = tk.BooleanVar()
                view_bbox.set(False)
                self.bboxes.append(view_bbox)
                self.bbox_opt.add_checkbutton(label=f" {self.class_names[self.labels[i].item()].capitalize()} ({i})", onvalue=1, offvalue=0, variable=self.bboxes[i], command=lambda idx=i: single_bbox_select(self, idx))

            # BBoxes initialization
            initialize_bboxes(self)


    def visualize(self):
        '''
        Visualizes predicted bounding boxes on all the cameras and shows
        the attention map in the middle of the plot.
        '''
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

        # Avoid selecting all layers and all cameras. Only the last layer will be visualized
        if self.show_all_layers.get() and self.selected_camera.get() == -1:
            self.selected_layer.set(self.Attention.layers - 1)

        # Extract the selected bounding box indexes from the menu
        self.bbox_idx = [i for i, x in enumerate(self.bboxes) if x.get()]

        # Extract Ground Truth bboxes if wanted
        if self.GT_bool.get():
            self.gt_bbox = self.dataloader.dataset.get_ann_info(self.data_idx)['gt_bboxes_3d']
        else:
            self.gt_bbox = None

        # Show attention map 
        if self.selected_expl_type.get() == "Attention Rollout" and self.selected_head_fusion.get() == "all":
            for k in range(len(self.head_types)):
                self.selected_head_fusion.set(self.head_types[k])
                self.show_attention_maps(grid_clm=k)
            self.selected_head_fusion.set("all")
        else:
            self.show_attention_maps()

        # Generate images list with bboxes on it
        print("Generating camera images...")
        self.cam_imgs = []
        og_imgs, bbox_cameras = [], []  # Used for video generation
        mask = np.zeros((3), dtype=np.uint8)
        org_shape = [900, 1600]
        for camidx in range(len(self.imgs)):
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
            
            og_img = cv2.cvtColor(self.imgs[camidx], cv2.COLOR_BGR2RGB)
            og_imgs.append(og_img)

            bbox_cameras.append(bbox_camera)

            if self.GT_bool.get():
                img, _, _ = draw_lidar_bbox3d_on_img(
                        self.gt_bbox,
                        img,
                        self.img_metas['lidar2img'][camidx],
                        self.img_metas,
                        color=(255, 0, 0),
                        mode_2d=self.bbox_2d.get())

            if self.overlay_bool.get():
                if self.selected_camera.get() == -1:
                    attn = self.attn_list[camidx]
                elif self.show_all_layers.get():
                    attn = self.attn_list[self.selected_layer.get()]
                else:
                    attn = self.attn_list[0]

                if (self.selected_camera.get() != -1 and camidx == self.selected_camera.get()) or (self.selected_camera.get() == -1):
                    img = overlay_attention_on_image(img, attn)

            if self.selected_camera.get() == -1:
                topk = self.topk_idx[camidx]
            elif self.show_all_layers.get():
                topk = self.topk_idx[self.selected_layer.get()]

            # img = torch.from_numpy(img.reshape(org_shape[0], org_shape[1], -1))
            # img = img.scatter_(-1, topk, 0)
            # # topk = [1, 1600x(900*1600)*d_r, 1] = indexes of top attentions
            # # img = [900, 1600, 3]
            # []
            # # scatter assigns zeros to 
            # img = img.reshape(*org_shape).numpy()
            img_t = torch.from_numpy(img)
            mask = torch.ones_like(img_t)
            for idx in topk:
                mask[idx[0], idx[1]] = 0

            img_masked = img_t * mask
            img = cv2.cvtColor(img_masked.numpy(), cv2.COLOR_BGR2RGB)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.cam_imgs.append(img)

        print("Done.\n")

        if self.gen_video_bool:
            return self.cam_imgs, og_imgs, bbox_cameras, self.labels

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
            capture(self)


