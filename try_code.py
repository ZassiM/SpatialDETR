import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

window = tk.Tk()

btn = tk.Label(window, text='A simple plot')
btn.grid(row=0, column=0, padx=20, pady=10)

x = ['Col A', 'Col B', 'Col C']

y = [50, 20, 80]

fig = plt.figure(figsize=(4, 5))
plt.bar(x=x, height=y)

# You can make your x axis labels vertical using the rotation
plt.xticks(x, rotation=90)

# specify the window as master
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().grid(row=1, column=0)

window.mainloop()