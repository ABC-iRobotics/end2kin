from __future__ import print_function
from sklearn import preprocessing
import numpy as np
import cv2 as cv
from math import sqrt
import sys
import scipy.io
from io import StringIO
import instrument_segmentation as segm
import tooltip_edge_finder as edgeFinder


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
    
    
def visualize_lab(lab_image):
    cv.imshow('lab', lab_image)
    return True

    
# Optical flow preprocessing for the NN - standardization
# flow: optical flow matrix
# returns with an optical flow matrix where the values are standardized

def standardize_flow(flow):
    flow_x_channel = flow[:,:,0]
    flow_y_channel = flow[:,:,1]

    flow_scaled_x_channel = preprocessing.scale(flow_x_channel)
    flow_scaled_y_channel = preprocessing.scale(flow_y_channel)

    flow_scaled_both_channel = np.array([flow_scaled_x_channel, flow_scaled_y_channel])
    flow_scaled_both_channel_transposed = np.transpose(flow_scaled_both_channel, (1,2,0))

    return flow_scaled_both_channel_transposed



# Optical flow preprocessing for the NN - normalization
# flow: optical flow matrix
# returns with an optical flow matrix where the values are normalized

def normalize_flow(flow):
    flow_x_channel = flow[:,:,0]
    flow_y_channel = flow[:,:,1]

    flow_min_x = flow_x_channel.min()
    flow_max_x = flow_x_channel.max()
    flow_normalized_x = (flow_x_channel - flow_min_x)/(flow_max_x - flow_min_x) 

    flow_min_y = flow_y_channel.min()
    flow_max_y = flow_y_channel.max()
    flow_normalized_y = (flow_y_channel - flow_min_y)/(flow_max_y - flow_min_y) 

    flow_normalized_both_channel = np.array([flow_normalized_x, flow_normalized_y])
    flow_normalized_both_channel_transposed = np.transpose(flow_normalized_both_channel, (1,2,0))

    return flow_normalized_both_channel_transposed



# Write the flow matrix to a mat file 
# matrix_name is the variable name to write
# out_filename is the name of the .mat file

def writeToFile_flow(matrix_name, out_filename):
    matfile = out_filename
    scipy.io.savemat(matfile, mdict={'out': matrix_name}, oned_as='row')
    matdata = scipy.io.loadmat(matfile)
    assert np.all(matrix_name == matdata['out'])


def histEq(img):
    equ = cv.equalizeHist(img)
    return equ
    


# Farneback optical flow calculation
# This function handles the video, calculates the optical flow and the normalized values of dx and dy. You can add here standardization.

# video filename: video input; use 0,1 or 2 if you want to use your own camera
# frame_num: number of frames of the whole video
# width: image width
# height: image height

# Next parameters are for the Farneback optical flow. 
# Recommended usage: prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0

# Returns with the 3D optical flow matrix (every row is the optical flow of one frame, which contains x and y values)

def optflow_main(video_filename, frame_num, width, height, pyr_scale = None, levels = 0.5, winsize = 3, iterations = 15, poly_n = 3, poly_sigma = 5, flags = 1.2, flow_p = 0):
    cap = get_cap(video_filename)
    prevgray = get_grayImage(cap)
    
    prevgray = histEq(prevgray)

    allFlow =  np.ndarray(shape = (frame_num, width*height, 2))


    for x in range(frame_num):
        rgb_cap = get_RGBImage(cap)
     
        gray = get_grayImage(cap)

        gray = histEq(gray)
        
        flow = cv.calcOpticalFlowFarneback(prevgray, gray, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags, flow_p)

        prevgray = gray

        optflow_visualized = visualize_flow(gray, flow)

        #instrument_segmentation = segm.mask_init(rgb_cap, flow)
        
        roi = segm.get_roi(rgb_cap, 100, 500, 100, 550)
        #print(roi.dtype)

        lab_mask = segm.lab_segmentation(roi)
        #print(lab_mask.dtype)
        lab_mask_binary = segm.get_lab_mask(lab_mask)
        lab_mask_binary_int = lab_mask_binary.astype(np.uint8)
        #print(lab_mask_binary_int.dtype)
        
        connected_l, connected_r = segm.undesired_objects(lab_mask_binary_int)
        #edges = edgeFinder.get_tool_edges(connected.astype(np.uint8), connected)
        
        corners_l = edgeFinder.get_corners(connected_l.astype(np.uint8), roi)

        #fitted_lines = edgeFinder.left_upper_lane(corners_l, roi)
        #visualizeEdges = visualize_lab(fitted_lines)
        roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2Lab)
        color_goodfeatures = edgeFinder.get_color_goodfeatures(corners_l, roi_hsv)
        sift = edgeFinder.get_sift(roi)
        img_points = edgeFinder.goodFeatures_clustering(corners_l, color_goodfeatures, roi)
        visualizeClusters = visualize_lab(img_points)

        if not optflow_visualized:
            break

    return allFlow # 3D optical flow matrix (every row is the optical flow of one frame, which contains x and y values)
