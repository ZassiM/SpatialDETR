
from PIL import ImageGrab
import os

def capture(self):
    x0 = self.winfo_rootx()
    y0 = self.winfo_rooty()
    x1 = x0 + self.canvas.get_width_height()[0]
    y1 = y0 + self.canvas.get_width_height()[1]
    
    im = ImageGrab.grab((x0, y0, x1, y1))
    self.suffix = 0
    path = f"screenshots/{self.model_name}_{self.data_idx}"

    if os.path.exists(path+"_"+str(self.suffix)+".png"):
        self.suffix += 1
    else:
        self.suffix = 0

    path += "_" + str(self.suffix) + ".png"
    im.save(path) # Can also say im.show() to display it