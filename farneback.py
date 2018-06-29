#!/usr/bin/env python

'''
Running...
(ESC  - exit)
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from scipy import signal
import cv2 as cv
import video  # modules video, tst_scene_render and common needed to capture/load image
import pdb
import numpy.linalg as lin

# First window -> Flow lines
def draw_flow(img, flow, step=8): # step adjusted to match dataset image
	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
	fx, fy = flow[y, x].T
	lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
	cv.polylines(vis, lines, 0, (0, 255, 0))
	for (x1, y1), (_x2, _y2) in lines:
		cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
	return vis

# Second window -> HSV colors
def draw_hsv(flow):
	h, w = flow.shape[:2]
	fx, fy = flow[:, :, 0], flow[:, :, 1]
	ang = np.arctan2(fy, fx) + np.pi
	v = np.sqrt(fx*fx+fy*fy)
	hsv = np.zeros((h, w, 3), np.uint8)
	hsv[..., 0] = ang*(180/np.pi/2)
	hsv[..., 1] = 255
	hsv[..., 2] = np.minimum(v*50, 255) # color intensity adjusted to match dataset image
	bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
	return bgr

def optFlowLK(currImg, nextImg, window):
	#  Get kernels on coord x, y and time
	kernel_x = np.array([[-1., 1.], [-1., 1.]])
	kernel_y = np.array([[-1., -1.], [1., 1.]])
	kernel_t = np.array([[1., 1.], [1., 1.]])  
	# Get center of window
	w = int(window/2 - 0.5)
	currImg = currImg / 255.  # normalize pixels
	nextImg = nextImg / 255.  # normalize pixels
	
	# Lucas Kanade Implementation
	
	# For each point, calculate I_x, I_y, I_t (partial derivatives)
	mode = 'same'
	fx = signal.convolve2d(currImg, kernel_x, boundary='symm', mode=mode)
	fy = signal.convolve2d(currImg, kernel_y, boundary='symm', mode=mode)
	ft = signal.convolve2d(nextImg, kernel_t, boundary='symm', mode=mode) + \
			signal.convolve2d(currImg, -kernel_t, boundary='symm', mode=mode)
	Vx = np.zeros(currImg.shape)
	Vy = np.zeros(currImg.shape)

	# Skip edges and calculate derivatives inside window
	for i in range(w, currImg.shape[0]-w):
		for j in range(w, currImg.shape[1]-w):
			Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()  # Flat because it represents each
			Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()  # pixel of the window centered on w
			It = ft[i-w:i+w+1, j-w:j+w+1].flatten()  # (pixels q1 to qn)
			
			# Get matrices of the equation
			A = np.vstack((Ix, Iy)).T
			B = It * -1
			
			# Get velocity vector
			try:
				vel = np.matrix((A.T).dot(A)).I.dot(A.T).dot(B).T
				Vx[i, j] = vel[0]
				Vy[i, j] = vel[1]
			except:  # If matrix is not invertible, skip this one
				pass

	flow = np.stack((Vx, Vy), axis=-1) # Return flow as single variable which goes to draw function

	return flow


if __name__ == '__main__':
	import sys
	print(__doc__)
	try:
		fn = sys.argv[1]
	except IndexError:
		fn = 0

	# Image sequence directory
	cam = cv.VideoCapture('Images/Filtered/Temporal/%03d.png', 0)
	ret, prev = cam.read()

	show_hsv = True # Draw colors on second window
	while True:
		img = cam.grab() # frameskip
		ret, img = cam.read()

		# Calculate Farneback
		flow = cv.calcOpticalFlowFarneback(prev, img, None, 0.6, 3, 15, 3, 5, 1.2, 0)
		prev = img

		cv.imshow('flow', draw_flow(img, flow))
		if show_hsv:
			cv.imshow('flow HSV', draw_hsv(flow))

		ch = cv.waitKey(1)
		if ch == 27:
			break
		if ch == ord('1'):
			show_hsv = not show_hsv
			print('HSV flow visualization is', ['off', 'on'][show_hsv])

	cam.release
	cv.destroyAllWindows()
