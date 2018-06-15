from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

files = [f for f in os.listdir('Images/tif')]
plt.ion()

plt.figure(figsize=(4, 3))
img = plt.imshow(plt.imread('Images/Filtered/Median/085.png'))
plt.axis('off')
while(1):
	for f in files:
		im = plt.imread('Images/Filtered/Median/' + f[:3] + '.png')
		img.set_data(im)
		plt.pause(.02)
		plt.draw()
