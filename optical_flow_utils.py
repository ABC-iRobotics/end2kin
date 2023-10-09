from __future__ import print_function
import numpy as np
import cv2 as cv
from math import sqrt
import sys
from io import StringIO

show_hsv = False # global variable for drawing optical flow

# Drawing optical flow
# The next 3 functions will be used in the visualize_flow function

def draw_flow(img, flow, step=6):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fhy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res


# Get video/camera images
# video_filename: the file name of the video what you want to use as an input for optical flow
# If you want to use your camera, the video_filename should be 0, 1 or 2

def get_cap(video_filename):
    cap = cv.VideoCapture(video_filename)
    global show_hsv
    show_hsv = False
    return cap


# BGR2GRAY (optical flow is working on grayscale images)

def get_grayImage(cap):
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray


def get_RGBImage(cap):
    ret, img = cap.read()
    return img
    

# Visualize optical flow

def visualize_flow(gray, flow):
    cv.imshow('flow', draw_flow(gray, flow))
    global show_hsv
    if show_hsv:
        cv.imshow('flow HSV', draw_hsv(flow))
    ch = cv.waitKey(5)
    if ch == 27:
        return False
    if ch == ord('1'):
        show_hsv = not show_hsv
        print('HSV flow visualization is', ['off', 'on'][show_hsv])
    return True
    
   
# Write the flow matrix to a mat file 
# matrix_name is the variable name to write
# out_filename is the name of the .mat file

def writeToFile_flow(matrix_name, out_filename):
    matfile = out_filename
    scipy.io.savemat(matfile, mdict={'out': matrix_name}, oned_as='row')
    matdata = scipy.io.loadmat(matfile)
    assert np.all(matrix_name == matdata['out'])
    
    
def flow_mean(flow):
    mean_x = np.mean(flow[:,:,0])
    mean_y = np.mean(flow[:,:,1])
    mean_x = np.true_divide(flow[:,:,1].sum(1),(flow[:,:,1]!=0).sum(1))
    mean_x = np.mean(mean_x)
    mean_y = np.true_divide(flow[:,:,0].sum(1),(flow[:,:,0]!=0).sum(1))
    mean_y = np.mean(mean_y)
    print(mean_x)
    return mean_x, mean_y



# Farneback optical flow calculation
# This function handles the video, calculates the optical flow and the normalized values of dx and dy. You can add here standardization.

# video filename: video input; use 0,1 or 2 if you want to use your own camera
# frame_num: number of frames of the whole video
# width: image width
# height: image height

# Next parameters are for the Farneback optical flow. 
# Recommended usage: prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0

# Returns with the 3D optical flow matrix (every row is the optical flow of one frame, which contains x and y values)

def optflow_main(prevgray, gray, pyr_scale = None, levels = 0.5, winsize = 3, iterations = 15, poly_n = 3, poly_sigma = 5, flags = 1.2, flow_p = 0):

    flow = cv.calcOpticalFlowFarneback(prevgray, gray, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags, flow_p)

    prevgray = gray

    #optflow_visualized = visualize_flow(gray, flow)

    #print(flow.shape)
    flowMean_x, flowMean_y = flow_mean(flow)
        
    return flow
