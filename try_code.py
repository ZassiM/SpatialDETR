# importing tkinter for gui
import tkinter as tk
 
# creating window
window = tk.Tk()
 
# setting attribute
window.state('zoomed')
window.title("Geeks For Geeks")
 
# creating text label to display on window screen
label = tk.Label(window, text="Hello Tkinter!")
label.pack()
 
window.mainloop()