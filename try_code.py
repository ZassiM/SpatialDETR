from tkinter import *

root = Tk()

list_items = ['item 1', 'item 2', 'item 3', 'item 4']
row_var = IntVar() 
column_vars=[]
count = 0

for item in list_items: 
    Radiobutton(text=item, variable=row_var, 
                value=item).grid(column=0,row=count) 
    var=IntVar()  ##unique in each for loop
    Radiobutton(text=item, variable=var, 
                value=item).grid(column=1, row=count)
    column_vars.append(var)
    count += 1
    


root.mainloop()