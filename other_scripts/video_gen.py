import cv2
import numpy as np
import glob
import os

frameSize = (500, 500)

frame = cv2.imread("./show_results/img_0.png")

# setting the frame width, height width 
# the width, height of first image 
height, width, _ = frame.shape

out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, (width, height))

i = 0
for filename in glob.glob('./show_results/*.png'):
    img = cv2.imread(f"./show_results/img_{i}.png")
    out.write(img)
    i+=1

out.release()