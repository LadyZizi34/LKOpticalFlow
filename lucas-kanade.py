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
import video
import pdb
import numpy.linalg as lin


def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    # pdb.set_trace()
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    # pdb.set_trace()
    hsv[..., 0] = ang*(180/np.pi/2)
    hsv[..., 1] = 255
    # hsv[...,2] = np.minimum(v*4, 255)
    # low_val_flags = v < 1e-2
    # v[low_val_flags] = 0
    hsv[..., 2] = np.minimum(v*50, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res


def myOptFlowLK(currImg, nextImg, window, tau=1e-2):
	kernel_x = np.array([[-1., 1.], [-1., 1.]])
	kernel_y = np.array([[-1., -1.], [1., 1.]])
	kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
	# window_size is odd, all the pixels with offset in between [-w, w] are inside the window
	w = int(window/2 - 0.5)
	currImg = currImg / 255.  # normalize pixels
	nextImg = nextImg / 255.  # normalize pixels
	# Implement Lucas Kanade
	# for each point, calculate I_x, I_y, I_t
	mode = 'same'
	fx = signal.convolve2d(currImg, kernel_x, boundary='symm', mode=mode)
	fy = signal.convolve2d(currImg, kernel_y, boundary='symm', mode=mode)
	ft = signal.convolve2d(nextImg, kernel_t, boundary='symm', mode=mode) + \
            signal.convolve2d(currImg, -kernel_t, boundary='symm', mode=mode)
	Vx = np.zeros(currImg.shape)
	Vy = np.zeros(currImg.shape)

	# within window (window * window), not considering edges
	# calculate partial derivatives of the image on x, y and t
	for i in range(w, currImg.shape[0]-w):
		for j in range(w, currImg.shape[1]-w):
			Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()  # flat because it represents each
			Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()  # pixel of the window centered on w
			It = ft[i-w:i+w+1, j-w:j+w+1].flatten()  # (pixels q1 to qn)
			# pdb.set_trace()
			A = np.vstack((Ix, Iy)).T
			B = It * -1
			# if threshold τ is larger than the smallest eigenvalue of A'A:
			try:
				vel = np.matrix((A.T).dot(A)).I.dot(A.T).dot(B).T
				Vx[i, j] = vel[0]
				Vy[i, j] = vel[1]
			except:  # not invertible, so skip this one
				pass

	flow = np.stack((Vx, Vy), axis=-1)

	return flow


if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    # cam = video.create_capture(fn)
    # cam = cv.VideoCapture('streetsequence_skip.mp4')
    # cam = cv.VideoCapture('trafficsequence.mpg')
    # integer width 3 with 0pad on left
    cam = cv.VideoCapture('Temporal/%03d.png', 0)
    # cam = cv.VideoCapture('Gaussian_Anomaly/%03d.png',0) # integer width 3 with 0pad on left
    # pdb.set_trace()
    ret, prev = cam.read()

    # prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    prevgray = prev

    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()

    show_hsv = True
    while True:
        ret = cam.grab()  # -> nao funciona com a video/camera pra pegar o farneback
        # ret = cam.grab()
        # ret = cam.grab()

        ret, img = cam.read()

        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = img

        # flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.6, 3, 15, 3, 5, 1.2, 0)
        flow = myOptFlowLK(prevgray, gray, 5)
        prevgray = gray

        cv.imshow('flow', draw_flow(gray, flow))
        if show_hsv:
            cv.imshow('flow HSV', draw_hsv(flow))
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv.imshow('glitch', cur_glitch)

        ch = cv.waitKey(1)
        # pdb.set_trace()
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])
    cam.release
    cv.destroyAllWindows()
