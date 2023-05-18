''' Functions used by the App interface. '''
import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import scrolledtext
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

def select_data_idx(self):
    popup = tk.Toplevel(self)
    popup.geometry("80x80")

    self.entry = tk.Entry(popup, width=20)
    self.entry.pack()

    button = tk.Button(popup, text="OK", command=lambda k=self:close_entry(k, popup))
    button.pack()

def close_entry(self, popup):
    idx = self.entry.get()
    if idx.isnumeric() and int(idx) <= (len(self.dataloader)-1):
        self.data_idx = int(idx)
        update_data_label(self)
        popup.destroy()
    else:
        show_message(self, f"Insert an integer between 0 and {len(self.dataloader)-1}")

def random_data_idx(self):
    idx = random.randint(0, len(self.dataloader)-1)
    self.data_idx = idx
    update_data_label(self)

def update_data_label(self):
    idx = self.data_idx
    info = f"Model: {self.model_name} | Dataloader: {self.dataloader_name} | Data index: {idx} | GPU ID: {self.gpu_id.get()}"
    self.info_text.set(info)

def update_thr(self):
    self.BB_bool.set(True)
    self.show_labels.set(True)

def single_bbox_select(self, idx):
    if self.single_bbox.get():
        for i in range(len(self.bboxes)):
            if i != idx:
                self.bboxes[i].set(False)

def select_all_bboxes(self):
    if hasattr(self, "bboxes"):
        for i in range(len(self.bboxes)):
            self.bboxes[i].set(True)

def update_scores(self):
    all_attentions = self.Attention.get_all_attn(self.bbox_idx, self.nms_idxs, self.selected_head_fusion.get(), self.selected_discard_ratio.get(), self.raw_attn.get())
    self.scores = []
    self.scores_perc = []
    if self.selected_layer.get() == -1:
        for layer in range(self.Attention.layers):
            attn = all_attentions[layer][self.selected_camera.get()]
            score = round(attn.sum().item(), 2)
            self.scores.append(score)
    else:
        for cam in range(len(all_attentions[self.selected_layer.get()])):
            attn = all_attentions[self.selected_layer.get()][cam]
            score = round(attn.sum().item(), 2)
            self.scores.append(score)

    sum_scores = sum(self.scores)
    if sum_scores > 0:
        for i in range(len(self.scores)):
            score_perc = round(((self.scores[i]/sum_scores)*100))
            self.scores_perc.append(score_perc)


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