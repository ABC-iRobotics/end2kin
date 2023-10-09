# Import the necessary libraries
import cv2
import numpy as np
import instrument_segmentation as segm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def get_grayImage(cap):
# Capture an image from 'cap' and convert it to grayscale
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def moving_average(x, w):
# Function to compute the moving average of an array 'x' with window size 'w'
    return np.convolve(x, np.ones(w), 'valid') / w

def skeletonization(img_g, filter_size):    
# Function to perform skeletonization on a grayscale image 'img_g' with a filter size 'filter_size'
        #img_g = get_grayImage(cap)
        h, w = img_g.shape[:2]

        # Threshold the image
        ret,img = cv2.threshold(img_g, 127, 255, 0)
        
        img = cv2.medianBlur(img, 9)

        # Step 1: Create an empty skeleton
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)

        # Get a Cross Shaped Kernel
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
        dist = cv2.distanceTransform(img,cv2.DIST_L2, cv2.DIST_MASK_3)
        #print(np.transpose(np.nonzero(dist)))
        ret1,thresh1 = cv2.threshold(dist, 0.1*dist.max(),255,cv2.THRESH_BINARY)
 
        # Normalized processing, otherwise the effect will not be seen
        norm = cv2.normalize(dist,dist,0,1,cv2.NORM_MINMAX)

        #transp=np.transpose(np.nonzero(norm))
        #print(transp.shape)
    
        # Repeat steps 2-4
        while True:
            #Step 2: Open the image
            open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            #Step 3: Substract open from the original image
            temp = cv2.subtract(img, open)
            #Step 4: Erode the original image and refine the skeleton
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()
            # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
            if cv2.countNonZero(img)==0:
                break
            # Define the structuring element
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

            # Apply the closing operation
            #closing = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel)
            #opening = cv2.morphologyEx(skel, cv2.MORPH_OPEN, kernel)
    
        filtered_skeleton = segm.undesired_objects_size(skel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(filter_size, filter_size))
        closing = cv2.morphologyEx(filtered_skeleton, cv2.MORPH_CLOSE, kernel)
        return closing, dist, thresh1, img, img_g
        
        
def peak_calc(closing, dist, thresh1, prominence_joint1): 
# Peak detection       
    radii = closing * dist
    transp_radii_everything = np.transpose(radii)
    transp_radii = np.transpose(np.nonzero(radii))
    
    # right tool
    xSorted_radii = transp_radii[np.argsort(transp_radii[:, 1]), :]

    xSorted_values = []
    sorted_index = []

    for i in range(len(xSorted_radii)):
        row, col = xSorted_radii[i]
        xSorted_values.append(radii[row,col])
        sorted_index.append(i)
        
    xSorted_values = np.asarray(xSorted_values)
    sorted_index = np.asarray(sorted_index)
    prev_val = xSorted_values[0]
    prev_val_all = []
    peaks_calc = []

    mean_xSorted = np.mean(xSorted_values)
    xsorted_splitted = np.array_split(xSorted_values, 3)

    tool_body = len(xSorted_values)//2
    tool_middle = len(xSorted_values)//4
    tool_tip = len(xSorted_values)//4
    splitted_0 = xSorted_values[0:tool_body]
    splitted_1 = xSorted_values[(tool_body+1):(tool_body+tool_middle)]
    splitted_2 = xSorted_values[(tool_body+tool_middle+1):(tool_body+tool_middle+tool_tip)]

    mean_split0 = np.mean(splitted_0)
    mean_split1 = np.mean(splitted_1)
    mean_split2 = np.mean(splitted_2)

    radii_peaks_body = []
    for i in range(0,len(splitted_0)):
        if(np.all(xSorted_values[i] < mean_split0)):
            peaks_calc.append(xSorted_values[i])
            radii_peaks_body.append(xSorted_radii[i,:])

    radii_peaks_middle = []
    range_middle = len(splitted_0)+len(splitted_1)
    for i in range(len(splitted_0),range_middle):
        if(np.all(xSorted_values[i] < mean_split1)):
            peaks_calc.append(xSorted_values[i])
            radii_peaks_middle.append(xSorted_radii[i,:])
                
    radii_peaks_tip = []
    range_tip = len(splitted_0)+len(splitted_1)+len(splitted_2)
    for i in range(range_middle, range_tip):
        if(np.all(xSorted_values[i] < mean_split2 + 10)):
            peaks_calc.append(xSorted_values[i])
            radii_tip = xSorted_radii[i,:]
            radii_peaks_tip.append(radii_tip)
           
    local_minima = argrelextrema(xSorted_values, np.less)
    local_minima = np.asarray(local_minima)

    peaks_calc = np.asarray(peaks_calc)
    local_minima_binary = transp_radii[local_minima]
    thresh = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)

    local_minima_binary = np.asarray(local_minima_binary)
    local_minima_binary = np.squeeze(local_minima_binary, axis=(0,))

    for peak in radii_peaks_body:
        y,x = peak.ravel()
        #cv2.circle(thresh,(x,y),radius = 3,color=(255, 0, 0),thickness=1)
            
    for peak_middle in radii_peaks_middle:
        y,x = peak_middle.ravel()
        #print(x,y)
        cv2.circle(thresh,(x,y),radius = 3,color=(0, 255, 0),thickness=1)
           
    for peak_tip in radii_peaks_tip:
        y,x = peak_tip.ravel()
        #print(x,y)
        cv2.circle(thresh,(x,y),radius = 3,color=(0, 0, 255),thickness=1) 
            
    for minima in local_minima_binary:
        y,x = minima.ravel()
        #print(x,y)
        #cv2.circle(thresh,(x,y),radius = 3,color=(255, 0, 0),thickness=1)  
        
    peaks2, _ = find_peaks(-xSorted_values, prominence = prominence_joint1) #threshold = 8)      # BEST!
    peaks2_binary = xSorted_radii[peaks2]
    for peak2 in peaks2_binary:
        y,x = peak2.ravel()
        #print(x,y)
        cv2.circle(thresh,(x,y),radius = 3,color=(255, 0, 0),thickness=1)  


    x = xSorted_values
    #x = moving_average(xSorted_values, 10)
    #peaks2, _ = find_peaks(-x, prominence=1)      # BEST!
    #peaks3, _ = find_peaks(-x, width=20)
    #peaks4, _ = find_peaks(-x, threshold=0.4)     # Required vertical distance to its direct neighbouring samples
    #plt.subplot(2, 2, 1)
    #plt.plot(peaks, x[peaks], "xr"); plt.plot(x); plt.legend(['distance'])
    #plt.subplot(2, 2, 2)
    #plt.plot(peaks2, x[peaks2], "ob"); plt.plot(x); plt.legend(['prominence'])
    #plt.subplot(2, 2, 3)
    #plt.plot(peaks3, x[peaks3], "vg"); plt.plot(x); plt.legend(['width'])
    #plt.subplot(2, 2, 4)
    #plt.plot(peaks4, x[peaks4], "xk"); plt.plot(x); plt.legend(['threshold'])
    #plt.show()
    #plt.plot(xSorted_moving) # plotting by columns
    #plt.show()

    cv2.imshow('Thresh',thresh)
    cv2.imshow('Frame',dist)
    cv2.imshow('Distance',radii)

    #cv2.imshow('Thresh',thresh)
    radii_peaks_tip = np.asarray(radii_peaks_tip)
    radii_peaks_middle = np.asarray(radii_peaks_middle)
    peaks2_binary = np.asarray(peaks2_binary)

    return radii_peaks_tip, peaks2_binary, radii_peaks_middle, thresh
    
    
def tip_peaks_number(radii_peaks_tip):
    return len(radii_peaks_tip)
    
def tool_isOpen_prev(radii_peaks_tip_prev, radii_peaks_tip):
    if(radii_peaks_tip_prev + 20 < radii_peaks_tip):
        isopen = 1
    else:
        isopen = 0
    return isopen
    
    
def tool_isOpen(radii_peaks_tip_prev, radii_peaks_tip):
    if(radii_peaks_tip_prev > radii_peaks_tip + 20):
        isopen = 0
    else:
        isopen = 1
    return isopen

def segments_clustering(radii_peaks_tip,img):
    h, w = img.shape[:2]

    position = radii_peaks_tip
    
    feature_set = np.empty((len(position), 2))
    for i in range(len(position)):
        feature_set[i,0] = position[i,1] 
        feature_set[i,1] = position[i,0] 
    
    Z = np.float32(feature_set)
  
    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now separate the data, Note the flatten()
    A = Z[label.ravel()==0]
    B = Z[label.ravel()==1]

    A = A[:,0:2]
    B = B[:,0:2]
    
    A = np.int0(A)
    for first_cluster in A:
        x,y = first_cluster.ravel()
        cv2.circle(img,(x,y),radius = 3,color=(0, 255, 0),thickness=-1)
        
    B = np.int0(B)
    for second_cluster in B:
        x,y = second_cluster.ravel()
        cv2.circle(img,(x,y),radius = 3,color=(0, 0, 255),thickness=-1)
        
    vx_1, vy_1, x_1, y_1  = cv2.fitLine(np.array(A),cv2.DIST_L2,0,0.01,0.01)
    line = [float(vx_1),float(vy_1),float(x_1),float(y_1)]
    left_pt_1 = int((-x_1*vy_1/vx_1) + y_1)
    right_pt_1 = int(((img.shape[1]-x_1)*vy_1/vx_1)+y_1)
    cv2.line(img,(img.shape[1]-1,right_pt_1),(0,left_pt_1),255,2)
         
    vx_2, vy_2, x_2, y_2  = cv2.fitLine(np.array(B),cv2.DIST_L2,0,0.01,0.01)
    line = [float(vx_2),float(vy_2),float(x_2),float(y_2)]
    left_pt_2 = int((-x_2*vy_2/vx_2) + y_2)
    right_pt_2 = int(((img.shape[1]-x_2)*vy_2/vx_2)+y_2)
    cv2.line(img,(img.shape[1]-1,right_pt_2),(0,left_pt_2),255,2)
    
    cv2.imshow('Thresh',img)
    
    return img
      
    
def tip_clustering(radii_peaks_tip,img):
    #img = img*255
    h, w = img.shape[:2]

    position = radii_peaks_tip
    
    feature_set = np.empty((len(position), 2))
    for i in range(len(position)):
        feature_set[i,0] = position[i,1] 
        feature_set[i,1] = position[i,0] 
    
    Z = np.float32(feature_set)
  
    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now separate the data, Note the flatten()
    A = Z[label.ravel()==0]
    B = Z[label.ravel()==1]

    A = A[:,0:2]
    B = B[:,0:2]
    
    A = np.int0(A)
    for first_cluster in A:
        x,y = first_cluster.ravel()
        cv2.circle(img,(x,y),radius = 3,color=(0, 255, 0),thickness=-1)
        
    B = np.int0(B)
    for second_cluster in B:
        x,y = second_cluster.ravel()
        cv2.circle(img,(x,y),radius = 3,color=(0, 0, 255),thickness=-1)
        
        
    vx_1, vy_1, x_1, y_1  = cv2.fitLine(np.array(A),cv2.DIST_L2,0,0.01,0.01)
    line = [float(vx_1),float(vy_1),float(x_1),float(y_1)]
    left_pt_1 = int((-x_1*vy_1/vx_1) + y_1)
    right_pt_1 = int(((img.shape[1]-x_1)*vy_1/vx_1)+y_1)
    cv2.line(img,(img.shape[1]-1,right_pt_1),(0,left_pt_1),255,2)
    
            
    vx_2, vy_2, x_2, y_2  = cv2.fitLine(np.array(B),cv2.DIST_L2,0,0.01,0.01)
    line = [float(vx_2),float(vy_2),float(x_2),float(y_2)]
    left_pt_2 = int((-x_2*vy_2/vx_2) + y_2)
    right_pt_2 = int(((img.shape[1]-x_2)*vy_2/vx_2)+y_2)
    cv2.line(img,(img.shape[1]-1,right_pt_2),(0,left_pt_2),255,2)
    
    
    cv2.imshow('Thresh',img)
    
    return img
    
    
def linefit_ifClosed(radii_peaks_tip,img):
       
    vy_1, vx_1, y_1, x_1  = cv2.fitLine(np.array(radii_peaks_tip),cv2.DIST_L2,0,0.01,0.01)
    line = [float(vx_1),float(vy_1),float(x_1),float(y_1)]
    left_pt_1 = int((-x_1*vy_1/vx_1) + y_1)
    right_pt_1 = int(((img.shape[1]-x_1)*vy_1/vx_1)+y_1)
    cv2.line(img,(img.shape[1]-1,right_pt_1),(0,left_pt_1),255,2)
    
    cv2.imshow('Thresh',img)
    
    return img
    

def findNearestContour(closing, dist, peaks2_binary, joint1, cap, c, prev_from_calc, prev_from_calc_upperB, prev_from_calc_lowerA, prev_from_calc_upperA, prev_from_calc_lastLowerB, prev_from_calc_lastUpperB,prev_from_calc_iLowerB, prev_from_calc_iUpperB, prev_from_calc_iLower, prev_from_calc_iUpper, prev_from_calc_lastiLowerB, prev_from_calc_lastiUpperB):
    
    if not peaks2_binary.any():
        return
    if(peaks2_binary.any()):
        firstPeakJointA = peaks2_binary[0]
    firstPeakWidth = dist[firstPeakJointA[0],firstPeakJointA[1]]
    peakRadius = firstPeakWidth/2
    h, w = dist.shape[:2]
    
    firstPeakJointB = joint1[0]
    lastPeakJointB = joint1[-1]
   
    img_g = get_grayImage(cap)
    ret, thresh = cv2.threshold(img_g, 127, 255, 0)
    thresh = cv2.medianBlur(thresh, 9)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    draw_contours = cv2.drawContours(thresh, contours, -1, (0,255,0), 3)
   
    yx_coords = np.column_stack(np.where((draw_contours[:,:,0] == 0) & (draw_contours[:,:,1] == 255) & (draw_contours[:,:,0] == 0)))
    contourPeakDistUpper = []
    contourPeakDistLower = []
    iUpper = []
    iLower = []

    for i in range(len(yx_coords)): #contour points to find the minimum distance
        if(yx_coords[i,0] > firstPeakJointA[0]): #location based on y coordinates
            contourPeakDistPointLower = np.linalg.norm(yx_coords[i] -firstPeakJointA)
            contourPeakDistLower.append(contourPeakDistPointLower) #distances of the lower edges
            iLower.append(i) #lower edge indices
        if(yx_coords[i,0] < firstPeakJointA[0]):
            contourPeakDistPointUpper = np.linalg.norm(yx_coords[i] -firstPeakJointA) 
            contourPeakDistUpper.append(contourPeakDistPointUpper) #distances of upper edges
            iUpper.append(i) #upper edge indices

    contourPeakDistUpperB = []
    contourPeakDistLowerB = []
    iUpperB = []
    iLowerB = []
    
    lastContourPeakDistUpperB = []
    lastContourPeakDistLowerB = []
    lastiUpperB = []
    lastiLowerB = []
    
    for i in range(len(yx_coords)): #contour points to find the minimum distance
        if(yx_coords[i,0] > firstPeakJointB[0]): #location based on y coordinates
            contourPeakDistPointLowerB = np.linalg.norm(yx_coords[i] -firstPeakJointB)
            contourPeakDistLowerB.append(contourPeakDistPointLowerB) #distances of the lower edges
            iLowerB.append(i) #lower edge indices
        if(yx_coords[i,0] < firstPeakJointB[0]):
            contourPeakDistPointUpperB = np.linalg.norm(yx_coords[i] -firstPeakJointB)
            contourPeakDistUpperB.append(contourPeakDistPointUpperB) #distances of upper edges
            iUpperB.append(i) #upper edge indices
    
    
    for i in range(len(yx_coords)): #contour points to find the minimum distance
        if(yx_coords[i,0] > lastPeakJointB[0]): #location based on y coordinates
            lastContourPeakDistPointLowerB = np.linalg.norm(yx_coords[i] -lastPeakJointB)
            lastContourPeakDistLowerB.append(lastContourPeakDistPointLowerB) #distances of the lower edges
            lastiLowerB.append(i) #lower edge indices
        if(yx_coords[i,0] < lastPeakJointB[0]):
            lastContourPeakDistPointUpperB = np.linalg.norm(yx_coords[i] -lastPeakJointB)
            lastContourPeakDistUpperB.append(lastContourPeakDistPointUpperB) #distances of upper edges
            lastiUpperB.append(i) #upper edge indices
    
    #######################################################################################
    if(c == 0):
        prev_contourPeakDistLowerB = contourPeakDistLowerB
        prev_contourPeakUpperB = contourPeakDistUpperB
        prev_lastContourPeakDistLowerB = lastContourPeakDistLowerB
        prev_lastContourPeakUpperB = lastContourPeakDistUpperB
        prev_contourPeakDistLower = contourPeakDistLower
        prev_contourPeakDistUpper = contourPeakDistUpper
        
        prev_iLowerB = iLowerB
        prev_iUpperB = iUpperB
        prev_iLower = iLower
        prev_iUpper = iUpper
        prev_lastiLowerB = lastiLowerB
        prev_lastiUpperB = lastiUpperB

    if(c > 0):
        prev_contourPeakDistLowerB = prev_from_calc
        prev_contourPeakDistUpperB = prev_from_calc_upperB
        prev_lastContourPeakDistLowerB = prev_from_calc_lastLowerB
        prev_lastContourPeakDistUpperB = prev_from_calc_lastUpperB
        prev_contourPeakDistLower = prev_from_calc_lowerA
        prev_contourPeakDistUpper = prev_from_calc_upperA
        
        prev_iLowerB = prev_from_calc_iLowerB
        prev_iUpperB = prev_from_calc_iUpperB
        prev_lastiLowerB = prev_from_calc_lastiLowerB
        prev_lastiUpperB = prev_from_calc_lastiUpperB
        prev_iLower = prev_from_calc_iLower
        prev_iUpper = prev_from_calc_iUpper
    
    if((c > 0) and (not contourPeakDistLowerB)):
        contourPeakDistLowerB = prev_contourPeakDistLowerB
        
    if((c > 0) and (not contourPeakDistUpperB)):
        contourPeakDistUpperB = prev_contourPeakDistUpperB
        
    if((c > 0) and (not contourPeakDistLower)):
        contourPeakDistLower = prev_contourPeakDistLower
        
    if((c > 0) and (not contourPeakDistUpper)):
        contourPeakDistUpper = prev_contourPeakDistUpper
    
    if((c > 0) and (not lastContourPeakDistLowerB)):
        lastContourPeakDistLowerB = prev_lastContourPeakDistLowerB
        
    if((c > 0) and (not lastContourPeakDistUpperB)):
        lastContourPeakDistUpperB = prev_lastContourPeakDistUpperB
            
    if((c > 0) and (not iLowerB)):
        iLowerB = prev_iLowerB
        
    if((c > 0) and (not iUpperB)):
        iUpperB = prev_iUpperB
        
    if((c > 0) and (not iLower)):
        iLower = prev_iLower
        
    if((c > 0) and (not iUpper)):
        iUpper = prev_iUpper
        
    if((c > 0) and (not lastiUpperB)):
        lastiUpperB = prev_lastiUpperB
        
    if((c > 0) and (not lastiLowerB)):
        lastiLowerB = prev_lastiLowerB
    #######################################################################################

    contourPeakDistLower = np.asarray(contourPeakDistLower)   
    contourPeakDistUpper = np.asarray(contourPeakDistUpper)  
    contourPeakDistLowerB = np.asarray(contourPeakDistLowerB)   
    contourPeakDistUpperB = np.asarray(contourPeakDistUpperB)  
    lastContourPeakDistLowerB = np.asarray(lastContourPeakDistLowerB)   
    lastContourPeakDistUpperB = np.asarray(lastContourPeakDistUpperB)  
    iUpper = np.asarray(iUpper)   
    iLower = np.asarray(iLower) 
    iUpperB = np.asarray(iUpperB)   
    iLowerB = np.asarray(iLowerB)   
    lastiUpperB = np.asarray(lastiUpperB)   
    lastiLowerB = np.asarray(lastiLowerB) 
    

    peakEdgeUpper = np.amin(contourPeakDistUpper)
    idxEdgeUpper = np.argmin(contourPeakDistUpper) # finding the index of the minimum value of the distances on the upper contour
    closestEdgePointToPeakUpper = iUpper[idxEdgeUpper]
    closestEdgePointToPeakUpper = yx_coords[closestEdgePointToPeakUpper]
      
    peakEdgeLower = np.amin(contourPeakDistLower)
    idxEdgeLower = np.argmin(contourPeakDistLower)
    closestEdgePointToPeakLower = iLower[idxEdgeLower]
    closestEdgePointToPeakLower = yx_coords[closestEdgePointToPeakLower]
    
    peakEdgeLowerB = np.amin(contourPeakDistLowerB)
    idxEdgeLowerB = np.argmin(contourPeakDistLowerB)
    closestEdgePointToPeakLowerB = iLowerB[idxEdgeLowerB]
    closestEdgePointToPeakLowerB = yx_coords[closestEdgePointToPeakLowerB]

    lastidxEdgeUpperB = np.argmin(lastContourPeakDistUpperB) # finding the index of the minimum value of the distances on the upper contour
    lastClosestEdgePointToPeakUpperB = lastiUpperB[lastidxEdgeUpperB]
    lastClosestEdgePointToPeakUpperB = yx_coords[lastClosestEdgePointToPeakUpperB]
    
    lastidxEdgeLowerB = np.argmin(lastContourPeakDistLowerB) # finding the index of the minimum value of the distances on the upper contour
    lastClosestEdgePointToPeakLowerB = lastiLowerB[lastidxEdgeLowerB]
    lastClosestEdgePointToPeakLowerB = yx_coords[lastClosestEdgePointToPeakLowerB]
    
    peakEdgeUpperB = np.amin(contourPeakDistUpperB)
    idxEdgeUpperB = np.argmin(contourPeakDistUpperB) # finding the index of the minimum value of the distances on the upper contour
    closestEdgePointToPeakUpperB = iUpperB[idxEdgeUpperB]
    closestEdgePointToPeakUpperB = yx_coords[closestEdgePointToPeakUpperB]

    firstPeakJointA = (closestEdgePointToPeakUpper + closestEdgePointToPeakLower) / 2
    firstPeakJointA = firstPeakJointA.astype(int)
    
    firstPeakJointB = (closestEdgePointToPeakUpperB + closestEdgePointToPeakLowerB) / 2
    firstPeakJointB = firstPeakJointB.astype(int)
    
    lastPeakJointB = (lastClosestEdgePointToPeakUpperB + lastClosestEdgePointToPeakLowerB) / 2
    lastPeakJointB = lastPeakJointB.astype(int)
    
    print(lastClosestEdgePointToPeakUpperB, lastClosestEdgePointToPeakLowerB, lastPeakJointB)
    cv2.circle(thresh,(closestEdgePointToPeakUpper[1],closestEdgePointToPeakUpper[0]),radius = 3,color=(255, 0, 0),thickness=1)  
    cv2.circle(thresh,(closestEdgePointToPeakLower[1],closestEdgePointToPeakLower[0]),radius = 3,color=(255, 0, 0),thickness=1) 
    cv2.circle(thresh,(closestEdgePointToPeakUpperB[1],closestEdgePointToPeakUpperB[0]),radius = 3,color=(255, 0, 0),thickness=1)  
    cv2.circle(thresh,(closestEdgePointToPeakLowerB[1],closestEdgePointToPeakLowerB[0]),radius = 3,color=(255, 0, 0),thickness=1) 
    cv2.circle(thresh,(lastClosestEdgePointToPeakUpperB[1],lastClosestEdgePointToPeakUpperB[0]),radius = 3,color=(0, 0, 255),thickness=1)  
    cv2.circle(thresh,(lastClosestEdgePointToPeakLowerB[1],lastClosestEdgePointToPeakLowerB[0]),radius = 3,color=(0, 0, 255),thickness=1)  
    cv2.circle(thresh,(firstPeakJointA[1],firstPeakJointA[0]),radius = 3,color=(255, 0, 0),thickness=1)  
    cv2.circle(thresh,(firstPeakJointB[1],firstPeakJointB[0]),radius = 3,color=(255, 0, 0),thickness=1)  
    cv2.circle(thresh,(lastPeakJointB[1],lastPeakJointB[0]),radius = 3,color=(0, 0, 255),thickness=1)  
    
    cv2.imshow('contours',thresh)
    
    #######################################################################################
    prev_from_calc = contourPeakDistLowerB
    prev_from_calc_upperB = contourPeakDistUpperB
    prev_from_calc_lastLowerB = lastContourPeakDistLowerB
    prev_from_calc_lastUpperB = lastContourPeakDistUpperB
    prev_from_calc_lowerA = contourPeakDistLower
    prev_from_calc_upperA = contourPeakDistUpper
    prev_from_calc_iLowerB = iLowerB
    prev_from_calc_iUpperB = iUpperB
    prev_from_calc_iLower = iLower
    prev_from_calc_iUpper = iUpper
    prev_from_calc_lastiLowerB = lastiLowerB
    prev_from_calc_lastiUpperB = lastiUpperB

    #######################################################################################
    
   # Pose based on segment 2 
    return firstPeakJointB, closestEdgePointToPeakUpperB, closestEdgePointToPeakLowerB, closestEdgePointToPeakUpper, firstPeakJointA, closestEdgePointToPeakLower, prev_from_calc, prev_from_calc_upperB, prev_from_calc_lowerA, prev_from_calc_upperA, prev_from_calc_lastLowerB, prev_from_calc_lastUpperB, prev_from_calc_iLowerB, prev_from_calc_iUpperB, prev_from_calc_iLower, prev_from_calc_iUpper, prev_from_calc_lastiLowerB, prev_from_calc_lastiUpperB

   # Pose based on segment 3
    #return firstPeakJointB, closestEdgePointToPeakUpperB, closestEdgePointToPeakLowerB, lastClosestEdgePointToPeakUpperB, lastPeakJointB, lastClosestEdgePointToPeakLowerB, prev_from_calc, prev_from_calc_upperB, prev_from_calc_lowerA, prev_from_calc_upperA, prev_from_calc_lastLowerB, prev_from_calc_lastUpperB, prev_from_calc_iLowerB, prev_from_calc_iUpperB, prev_from_calc_iLower, prev_from_calc_iUpper, prev_from_calc_lastiLowerB, prev_from_calc_lastiUpperB

def findNearestContour_peakFound(closing, dist, peaks2_binary, joint1, cap):
    
    if not peaks2_binary.any():
        return
    if(peaks2_binary.any()):
        firstPeakJointA = peaks2_binary[0]
    firstPeakWidth = dist[firstPeakJointA[0],firstPeakJointA[1]]
    peakRadius = firstPeakWidth/2
    h, w = dist.shape[:2]
    
    firstPeakJointB = joint1[0]
   
    img_g = get_grayImage(cap)
    ret, thresh = cv2.threshold(img_g, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    draw_contours = cv2.drawContours(thresh, contours, -1, (0,255,0), 3)

        
    yx_coords = np.column_stack(np.where((draw_contours[:,:,0] == 0) & (draw_contours[:,:,1] == 255) & (draw_contours[:,:,0] == 0)))
    contourPeakDistUpper = []
    contourPeakDistLower = []
    iUpper = []
    iLower = []

    for i in range(len(yx_coords)): #contour points to find the minimum distance
        if(yx_coords[i,0] > firstPeakJointA[0]): #location based on y coordinates
            contourPeakDistPointLower = np.linalg.norm(yx_coords[i] -firstPeakJointA)
            contourPeakDistLower.append(contourPeakDistPointLower) #distances of the lower edges
            iLower.append(i) #lower edge indices
        if(yx_coords[i,0] < firstPeakJointA[0]):
            contourPeakDistPointUpper = np.linalg.norm(yx_coords[i] -firstPeakJointA) 
            contourPeakDistUpper.append(contourPeakDistPointUpper) #distances of upper edges
            iUpper.append(i) #upper edge indices

    contourPeakDistUpperB = []
    contourPeakDistLowerB = []
    iUpperB = []
    iLowerB = []
    
    
    for i in range(len(yx_coords)): #contour points to find the minimum distance
        if(yx_coords[i,0] > firstPeakJointB[0]): #location based on y coordinates
            contourPeakDistPointLowerB = np.linalg.norm(yx_coords[i] -firstPeakJointB)
            contourPeakDistLowerB.append(contourPeakDistPointLowerB) #distances of the lower edges
            iLowerB.append(i) #lower edge indices
        if(yx_coords[i,0] < firstPeakJointB[0]):
            contourPeakDistPointUpperB = np.linalg.norm(yx_coords[i] -firstPeakJointB)
            contourPeakDistUpperB.append(contourPeakDistPointUpperB) #distances of upper edges
            iUpperB.append(i) #upper edge indices
    
    contourPeakDistLower = np.asarray(contourPeakDistLower)   
    contourPeakDistUpper = np.asarray(contourPeakDistUpper)  
    contourPeakDistLowerB = np.asarray(contourPeakDistLowerB)   
    contourPeakDistUpperB = np.asarray(contourPeakDistUpperB)  
    iUpper = np.asarray(iUpper)   
    iLower = np.asarray(iLower) 
    iUpperB = np.asarray(iUpperB)   
    iLowerB = np.asarray(iLowerB)   

    peakEdgeUpper = np.amin(contourPeakDistUpper)
    idxEdgeUpper = np.argmin(contourPeakDistUpper) # finding the index of the minimum value of the distances on the upper contour
    closestEdgePointToPeakUpper = iUpper[idxEdgeUpper]
    closestEdgePointToPeakUpper = yx_coords[closestEdgePointToPeakUpper]
    
    peakEdgeLower = np.amin(contourPeakDistLower)
    idxEdgeLower = np.argmin(contourPeakDistLower)
    closestEdgePointToPeakLower = iLower[idxEdgeLower]
    closestEdgePointToPeakLower = yx_coords[closestEdgePointToPeakLower]
    
    peakEdgeLowerB = np.amin(contourPeakDistLowerB)
    idxEdgeLowerB = np.argmin(contourPeakDistLowerB)
    closestEdgePointToPeakLowerB = iLowerB[idxEdgeLowerB]
    closestEdgePointToPeakLowerB = yx_coords[closestEdgePointToPeakLowerB]

    peakEdgeUpperB = np.amin(contourPeakDistUpperB)
    idxEdgeUpperB = np.argmin(contourPeakDistUpperB) # finding the index of the minimum value of the distances on the upper contour
    closestEdgePointToPeakUpperB = iUpperB[idxEdgeUpperB]
    closestEdgePointToPeakUpperB = yx_coords[closestEdgePointToPeakUpperB]
   
    firstPeakJointA = (closestEdgePointToPeakUpper + closestEdgePointToPeakLower) / 2
    firstPeakJointA = firstPeakJointA.astype(int)
    
    firstPeakJointB = (closestEdgePointToPeakUpperB + closestEdgePointToPeakLowerB) / 2
    firstPeakJointB = firstPeakJointB.astype(int)
    
    cv2.circle(thresh,(closestEdgePointToPeakUpper[1],closestEdgePointToPeakUpper[0]),radius = 3,color=(255, 0, 0),thickness=1)  
    cv2.circle(thresh,(closestEdgePointToPeakLower[1],closestEdgePointToPeakLower[0]),radius = 3,color=(255, 0, 0),thickness=1) 
    cv2.circle(thresh,(closestEdgePointToPeakUpperB[1],closestEdgePointToPeakUpperB[0]),radius = 3,color=(255, 0, 0),thickness=1)  
    cv2.circle(thresh,(closestEdgePointToPeakLowerB[1],closestEdgePointToPeakLowerB[0]),radius = 3,color=(255, 0, 0),thickness=1)  
    cv2.circle(thresh,(firstPeakJointA[1],firstPeakJointA[0]),radius = 3,color=(255, 0, 0),thickness=1)  
    cv2.circle(thresh,(firstPeakJointB[1],firstPeakJointB[0]),radius = 3,color=(255, 0, 0),thickness=1)  
    
    cv2.imshow('contours',thresh)
    
    return firstPeakJointB, closestEdgePointToPeakUpperB, closestEdgePointToPeakLowerB, closestEdgePointToPeakUpper, firstPeakJointA, closestEdgePointToPeakLower
    
    

def plotter(point):
    fig = plt.figure()
 
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
 
    # defining all 3 axes
    z = np.linspace(0, 1, 100)
    x = point[0]
    y = point[1]
 
    # plotting
    ax.plot3D(x, y, z, 'green')
    ax.set_title('3D line plot geeks for geeks')
    plt.show()
    plt.pause(0.0001)           
    
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
  
