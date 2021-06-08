from skimage.data import camera
from skimage.filters import frangi, hessian
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np
from matplotlib.colors import hsv_to_rgb
import numpy as np


def get_moving_pixel_indices(flow):
    h, w = flow.shape[:2]
    
    mask_moving_pixels = np.empty((h, w))
    
    for i in range(h):
        for j in range(w):
            if (flow[i,j,0] > 10 or flow[i,j,1] > 10 or flow[i,j,0] < -10 or flow[i,j,1] < -10):
                mask_moving_pixels[i,j] = 1
            else:
                mask_moving_pixels[i,j] = 0
    #print(flow)
    return mask_moving_pixels


def lab_segmentation(image):

    lowerRange= np.array([50, 120, 120] , dtype="uint8")
    upperRange= np.array([120, 140, 140], dtype="uint8")
    
    lowerRange_tool= np.array([0, 120, 120] , dtype="uint8")
    upperRange_tool= np.array([50, 140, 140], dtype="uint8")
    
    mask = image[:].copy()
    mask_tool = image[:].copy()
    
    imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    imageRange = cv2.inRange(imageLab,lowerRange, upperRange)
    imageRange_tool = cv2.inRange(imageLab,lowerRange_tool, upperRange_tool)
    
    mask[:,:,0] = imageRange
    mask[:,:,1] = imageRange
    mask[:,:,2] = imageRange
    
    mask_tool[:,:,0] = imageRange_tool
    mask_tool[:,:,1] = imageRange_tool
    mask_tool[:,:,2] = imageRange_tool
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    faceLab = cv2.bitwise_and(image,mask)
    faceLab_tool = cv2.bitwise_and(image,mask_tool)
    
    faceLab_all = mask_tool + mask
    
    return faceLab_all
    

def get_lab_mask(faceLab_all):
    h, w = faceLab_all.shape[:2]
    
    mask_lab = np.empty((h, w))
    
    for i in range(h):
        for j in range(w):
            if (faceLab_all[i,j,0] > 0):
                mask_lab[i,j] = 1
            else:
                mask_lab[i,j] = 0
    #print(flow)
    return mask_lab



def probabilistic_Hough(lab_mask):

    dst = cv2.Canny(lab_mask, 50., 200., 3)
    color_dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(dst, rho=1., theta=np.pi/180.,
                        threshold=80, minLineLength=30, maxLineGap=10.)
    for this_line in lines:
        cv2.line(color_dst,
                (this_line[0][0], this_line[0][1]),
                (this_line[0][2], this_line[0][3]),
                [0, 0, 255], 3, 8)
    
    return color_dst
    
    
    
def get_flow_lab_mask(lab_mask, mask_moving_pixels):
    h, w = lab_mask.shape[:2]
    flow_lab_mask = np.empty((h, w))
    
    for i in range(h):
        for j in range(w):
            if (lab_mask[i,j] == 0):
                flow_lab_mask[i,j] = 0
            elif (lab_mask[i,j] == 1):
                if (mask_moving_pixels[i,j] == 1):
                    flow_lab_mask[i,j] = 1
                    #print(flow_lab_mask[i,j])
            else:
                print("lab mask is not a binary image!")
            

    return flow_lab_mask

    
def get_roi(img, range_1, range_2, range_3, range_4):
    roi = img[range_1:range_2, range_3:range_4]
    return roi
    
def mask_init(image, flow):
    flow_mask = get_moving_pixel_indices(flow)
    lab_segm = lab_segmentation(image)
    lab_mask = get_lab_mask(lab_segm)
    flow_lab_mask = get_flow_lab_mask(lab_mask, flow_mask)
    return flow_lab_mask



def connected_components(roi):
    #img = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    num_labels, labels_im = cv2.connectedComponents(roi)
    
    # Map component labels to hue val
    label_hue = np.uint8(179*labels_im/np.max(labels_im))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img



def undesired_objects(image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    
    sorted_indices = np.argsort(stats[:,-1])[::-1]
    
    max_area_index = sorted_indices[1]
    next_max_area_index = sorted_indices[2]
    
    max_size = sizes[max_area_index]
    max_size = sizes[next_max_area_index]
    
    (center_x_0, center_y_0) = centroids[max_area_index]
    (center_x_1, center_y_1) = centroids[next_max_area_index]

    left_tool = np.zeros(output.shape)
    right_tool = np.zeros(output.shape)
    
    if (center_x_0 < center_x_1):
        max_area_index = sorted_indices[1]
        next_max_area_index = sorted_indices[2]
    else:
        max_area_index = sorted_indices[2]
        next_max_area_index = sorted_indices[1]

    left_tool[output == max_area_index] = 255
    right_tool[output == next_max_area_index] = 255
    return left_tool, right_tool


























    

