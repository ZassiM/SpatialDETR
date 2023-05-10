import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import scrolledtext
import random


def show_info(self):
    if not self.showed_info.get():
        self.info_label.pack(side=tk.TOP)
        self.showed_info.set(True)
    else:
        self.info_label.pack_forget()
        self.showed_info.set(False)

def show_message(self, message):
    showinfo(title=None, message=message)

def red_text(self, event=None):
    self.info_label.config(fg="red")

def black_text(self, event=None):
    self.info_label.config(fg="black")

def show_model_info(self, event=None):
    popup = tk.Toplevel(self)
    popup.geometry("700x1000")
    popup.title(f"Model {self.model_name.capitalize()}")

    text = scrolledtext.ScrolledText(popup, wrap=tk.WORD)
    for k, v in self.model.module.__dict__["_modules"].items():
        text.insert(tk.END, f"{k.upper()}\n", 'key')
        text.insert(tk.END, f"{v}\n\n")
        text.tag_config('key', background="yellow", foreground="red")

    text.pack(expand=True, fill='both')
    text.configure(state="disabled")

def select_data_idx(self):
    popup = tk.Toplevel(self)
    popup.geometry("50x50")

    self.entry = tk.Entry(popup, width=20)
    self.entry.pack()

    button = tk.Button(popup, text="OK", command=lambda k=self:close_entry(k, popup))
    button.pack()

def close_entry(self, popup):
    idx = self.entry.get()
    if idx.isnumeric() and int(idx) <= (len(self.data_loader)-1):
        self.data_idx = int(idx)
        update_data_label(self, int(idx))
        popup.destroy()
    else:
        show_message(self, f"Insert an integer between 0 and {len(self.data_loader)-1}")


def update_data_label(self, idx):
    info = f"Model name: {self.model_name} | GPU ID: {self.gpu_id.get()} | Data index: {int(idx)}"
    self.info_text.set(info)
    

def random_data_idx(self):
    idx = random.randint(0, len(self.data_loader)-1)
    self.data_idx = idx
    update_data_label(self, idx)
    
def update_thr(self):
    self.BB_bool.set(True)
    self.show_labels.set(True)
    