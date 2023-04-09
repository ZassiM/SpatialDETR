#Import the tkinter library
from tkinter import *

#Create an instance of tkinter frame
win = Tk()

#Set the geometry
win.geometry("650x250")

#Add a text label and add the font property to it
label= Label(win, text= "Hello World!", font=('Times New Roman bold',20))
label.pack(padx=10, pady=10)

#Create a fullscreen window
win.attributes('-fullscreen', True)

win.mainloop()