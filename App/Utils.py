''' Functions used by the App interface. '''
import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import scrolledtext
import cv2
import numpy as np
import random
import os
from PIL import ImageGrab


def add_separator(self, sep="|"):
    self.menubar.add_command(label=sep, activebackground=self.menubar.cget("background"))
    # sep="\u22EE"


def show_message(self, message):
    showinfo(title=None, message=message)


def red_text(self, event=None):
    self.info_label.config(fg="red")


def black_text(self, event=None):
    if self.tk.call("ttk::style", "theme", "use") == "azure-dark":
        self.info_label.config(fg="white")
    else:
        self.info_label.config(fg="black")


def show_model_info(self, event=None):
    popup = tk.Toplevel(self)
    popup.geometry("700x1000")
    popup.title(f"Model {self.model_name}")

    text = scrolledtext.ScrolledText(popup, wrap=tk.WORD)
    for k, v in self.model.module.__dict__["_modules"].items():
        text.insert(tk.END, f"{k.upper()}\n", 'key')
        text.insert(tk.END, f"{v}\n\n")
        text.tag_config('key', background="yellow", foreground="red")

    text.pack(expand=True, fill='both')
    text.configure(state="disabled")


def select_data_idx(self, length=False):
    popup = tk.Toplevel(self)
    popup.geometry("80x50")

    self.entry = tk.Entry(popup, width=20)
    self.entry.pack()

    button = tk.Button(popup, text="OK", command=lambda k=self: close_entry(k, popup, length))
    button.pack()

def close_entry(self, popup, length):
    idx = self.entry.get()
    if not length:
        if idx.isnumeric() and int(idx) <= (len(self.dataloader)-1):
            self.data_idx = int(idx)
            update_info_label(self)
            popup.destroy()
        else:
            show_message(self, f"Insert an integer between 0 and {len(self.dataloader)-1}")
    else:
        if idx.isnumeric() and int(idx) <= ((len(self.dataloader)-1) - self.data_idx):
            self.video_length = int(idx)
            update_info_label(self)
            popup.destroy()
        else:
            show_message(self, f"Insert an integer between 0 and {(len(self.dataloader)-1) - self.data_idx}")       


def random_data_idx(self):
    idx = random.randint(0, len(self.dataloader)-1)
    self.data_idx = idx
    update_info_label(self)


def update_info_label(self, info=None, idx=None):
    if idx is None:
        idx = self.data_idx
    if info is None:
        info = f"Model: {self.model_name} | Dataloader: {self.dataloader_name} | Data index: {idx} | Mechanism: {self.selected_expl_type.get()}"
        if self.selected_camera.get() != -1 and not self.show_all_layers.get():
            info += f" | Camera {list(self.cameras.keys())[self.selected_camera.get()]} | Layer {self.selected_layer.get()}"
            if self.selected_expl_type.get() == "Attention Rollout":
                info += f'| {self.selected_head_fusion.get().capitalize()} Head fusion'
    self.info_text.set(info)


def update_thr(self):
    self.BB_bool.set(True)
    self.show_labels.set(True)


def single_bbox_select(self, idx):
    if self.single_bbox.get():
        for i in range(len(self.bboxes)):
            if i != idx:
                self.bboxes[i].set(False)


def initialize_bboxes(self):
    if hasattr(self, "bboxes"):
        if self.select_all_bboxes.get():
            for i in range(len(self.bboxes)):
                self.bboxes[i].set(True)
        else:
            if len(self.bboxes) > 0:
                self.bboxes[0].set(True)


def update_scores(self):
    all_attentions = self.Attention.get_all_attn(self.bbox_idx, self.nms_idxs, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get())
    scores = []
    self.scores_perc = []
    if self.show_all_layers.get():
        for layer in range(self.Attention.layers):
            attn = all_attentions[layer][self.selected_camera.get()]
            score = round(attn.sum().item(), 2)
            scores.append(score)
    else:
        for cam in range(len(all_attentions[self.selected_layer.get()])):
            attn = all_attentions[self.selected_layer.get()][cam]
            score = round(attn.sum().item(), 2)
            scores.append(score)

    sum_scores = sum(scores)
    if sum_scores > 0:
        for i in range(len(scores)):
            score_perc = round(((scores[i]/sum_scores)*100))
            self.scores_perc.append(score_perc)


def overlay_attention_on_image(img, attn):
    attn = cv2.applyColorMap(np.uint8(255 * attn), cv2.COLORMAP_JET)
    attn = np.float32(attn) 
    attn = cv2.resize(attn, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    img = attn + np.float32(img)
    img = img / np.max(img)
    return img


def capture(self):
    x0 = self.winfo_rootx()
    y0 = self.winfo_rooty()
    x1 = x0 + self.canvas.get_width_height()[0]
    y1 = y0 + self.canvas.get_width_height()[1]
    
    im = ImageGrab.grab((x0, y0, x1, y1))
    path = f"screenshots/{self.model_name}_{self.data_idx}"

    if os.path.exists(path+"_"+str(self.file_suffix)+".png"):
        self.file_suffix += 1
    else:
        self.file_suffix = 0

    path += "_" + str(self.file_suffix) + ".png"
    im.save(path)


def change_theme(self):
    if self.dark_theme.get():
        # Set light theme
        self.tk.call("set_theme", "dark")
    else:
        # Set dark theme
        self.tk.call("set_theme", "light")