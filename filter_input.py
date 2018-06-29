from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import time

def saveAsTempMedianFilter(imgList, n):
	imgsOut = np.zeros(imgList.shape, dtype=float)
	# Central index of filter, equivalent to padding needed
	center = int(n/2 - 0.5)
	imgList = np.pad(imgList, ((center,), (0,), (0,)), 'edge') # Padding on the "time"

	for i in range(center, imgList.shape[0] - center):
		iOut = i - center
		# Get median of neighbours
		neighbourhood = imgList[i-center:i+center+1]  
		median = np.median(neighbourhood, axis=0)  
		# Normalize
		imax = np.max(median)
		imin = np.min(median)
		minRange = 255 * imin / imax
		imgNorm = (median - imin)/(imax - imin)
		imgNorm = minRange + (imgNorm*(255-minRange))
		imgNorm = imgNorm.astype(np.uint8)  
		median = imgNorm
		imgsOut[iOut] = median
		# Save new image
		Image.fromarray(median).save('Images/Filtered/Temporal/' + '%03d' % (iOut+1) + '.png')

	return imgsOut

def gaussianFilter(img, n, sigma):
	imgOut = np.zeros(img.shape, dtype=np.complex)

	def gaussian_kernel(n, sigma):
		kernel_2d = np.zeros([n, n], dtype=float)
		center = int(n/2 - 0.5)

		x2d, y2d = np.meshgrid(np.linspace(-center, center, n),
		                       np.linspace(-center, center, n), sparse=True)
		kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
		kernel_2d = kernel_2d / (2 * np.pi * sigma ** 2)

		norm_kernel_2d = kernel_2d/np.sum(kernel_2d)

		return norm_kernel_2d

	# Convolution between image f and filter w
	def convolution_2d(f, w):
		central_index_x = int(w.shape[0]/2 - 0.5)
		central_index_y = int(w.shape[1]/2 - 0.5)
		g = np.zeros(f.shape, dtype=float)
		
		f = np.pad(f, ((central_index_x,), (central_index_y,)), 'edge')
		for x in range(g.shape[0]):
			for y in range(g.shape[1]):
				sub_f = f[x: x+w.shape[0], y: y+w.shape[1]]
				g[x, y] = np.sum(np.multiply(sub_f, w))

		return g

	kernel = gaussian_kernel(n,sigma)
	imgOut = convolution_2d(img, kernel)

	# Normalize
	imax = np.max(imgOut)
	imin = np.min(imgOut)
	minRange = 255 * imin / imax
	imgNorm = (imgOut - imin)/(imax - imin)
	imgNorm = minRange + (imgNorm*(255-minRange))
	imgNorm = imgNorm.astype(np.uint8)  
	imgOut = imgNorm
	
	return imgOut

files = [f for f in os.listdir('Images/tif/long/')]
plt.ion()

filtered_frames = np.zeros([len(files), 240, 360], dtype=object)
index = 0
for f in files:
	im = Image.open('Images/tif/long/' + f)

	# convert to numpy array
	img = np.array(im)

	# apply some filter with window size 5
	filtered_img = gaussianFilter(img, 5, 1)
	filtered_frames[index] = filtered_img
	index += 1

plt.ioff()
# filtered_frames = tempMedianFilter(filtered_frames, 5)
saveAsTempMedianFilter(filtered_frames, 5)






