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
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from pandas import DataFrame




def get_tool_edges(mask, img):

    mask = mask*255
    #print(mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    epsilon = 0.1 * cv2.arcLength(contours[0], True)

    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    contours = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    
    return contours


def get_corners(mask, img):
    mask = mask*255
    
    mask = np.float32(mask)
    
    corners = cv2.goodFeaturesToTrack(mask, 100, 0.01, 10)
    corners = np.int0(corners)
    
    for corner in corners:
        x,y = corner.ravel()
        #cv2.circle(img,(x,y),3,255,-1)
  
    return corners
    

def left_upper_lane(corners_l, img):
    h, w = img.shape[:2]
    
    #print(corners_l.shape)
    corners_x_np = np.asarray(corners_l[:,:,0])
    sorted_x = corners_x_np.argsort(axis=0)
    sorted_x_max = sorted_x[0]
    
    corners_y_np = np.asarray(corners_l[:,:,1])
    sorted_y = corners_y_np.argsort(axis=0)
    sorted_y_max = sorted_y[0]
    
    
    # sort the points based on their x-coordinates

    corners_l = corners_l[:, 0]

    xSorted = corners_l[np.argsort(corners_l[:, 0]), :]

	# grab the left-most and right-most points from the sorted
	# x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    #print(tl)
    #print(bl)
    
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]

    br_tr = rightMost[np.argsort(D)[::-1], :]
    br = br_tr[0]
    tr = br_tr[1]
    
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
    critical_points = np.array([tl, tr, br, bl])
    lane_fit_lu = np.array([tl, tr])
    lane_fit_ld = np.array([bl, br])
    
    #print(critical_points)
    
    vx, vy, x, y  = cv2.fitLine(np.array(lane_fit_lu),cv2.DIST_L2,0,0.01,0.01)
    line = [float(vx),float(vy),float(x),float(y)]


    left_pt = int((-x*vy/vx) + y)
    right_pt = int(((img.shape[1]-x)*vy/vx)+y)
    cv2.line(img,(img.shape[1]-1,right_pt),(0,left_pt),255,2)

		
    return img
    
    
def goodFeatures_clustering(goodfeatures, color_goodfeatures, img):

    h, w = img.shape[:2]

    position = goodfeatures
    #print(position.shape)
    color = np.int64(color_goodfeatures)
    #print(color_goodfeatures.shape)
    
    feature_set = np.empty((len(color), 5))
    for i in range(len(color)):
        feature_set[i,0] = goodfeatures[i,:,0] 
        feature_set[i,1] = goodfeatures[i,:,1] 
        feature_set[i,2] = color_goodfeatures[i,:,0] 
        feature_set[i,3] = color_goodfeatures[i,:,1] 
        feature_set[i,4] = color_goodfeatures[i,:,2] 
    
    
    #print(feature_set.shape)
    
    
    
    Z = np.float32(feature_set)
    #print(Z)
    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now separate the data, Note the flatten()
    A = Z[label.ravel()==0]
    B = Z[label.ravel()==1]
    C = Z[label.ravel()==2]
    #D = Z[label.ravel()==3]
    # Plot the data
    #print(A)
    #A = A.reshape(-1,2)
    #B = B.reshape(-1,2)
    #C = C.reshape(-1,2)
    #D = D.reshape(-1,2)
    #print(A)
    #print(B.shape)

    A = A[:,0:2]
    B = B[:,0:2]
    C = C[:,0:2]
    #print(A)
    A = np.int0(A)
    for first_cluster in A:
        x,y = first_cluster.ravel()
        #cv2.circle(img,(x,y),radius = 3,color=(0, 255, 0),thickness=-1)
        
    B = np.int0(B)
    for second_cluster in B:
        x,y = second_cluster.ravel()
        #cv2.circle(img,(x,y),radius = 3,color=(0, 0, 255),thickness=-1)
        
    C = np.int0(C)
    for third_cluster in C:
        x,y = third_cluster.ravel()
        #cv2.circle(img,(x,y),radius = 3,color=(255, 0, 0),thickness=-1)
        
    #D = np.int0(D)
    #for fourth_cluster in D:
    #    x,y = fourth_cluster.ravel()
    #    cv2.circle(img,(x,y),radius = 3,color=(255, 255, 0),thickness=-1)
        
        
    vx_1, vy_1, x_1, y_1  = cv2.fitLine(np.array(A),cv2.DIST_L2,0,0.01,0.01)
    line = [float(vx_1),float(vy_1),float(x_1),float(y_1)]
    left_pt_1 = int((-x_1*vy_1/vx_1) + y_1)
    right_pt_1 = int(((img.shape[1]-x_1)*vy_1/vx_1)+y_1)
    #cv2.line(img,(img.shape[1]-1,right_pt_1),(0,left_pt_1),255,2)
    
            
    vx_2, vy_2, x_2, y_2  = cv2.fitLine(np.array(B),cv2.DIST_L2,0,0.01,0.01)
    line = [float(vx_2),float(vy_2),float(x_2),float(y_2)]
    left_pt_2 = int((-x_2*vy_2/vx_2) + y_2)
    right_pt_2 = int(((img.shape[1]-x_2)*vy_2/vx_2)+y_2)
    #cv2.line(img,(img.shape[1]-1,right_pt_2),(0,left_pt_2),255,2)
    
            
    vx_3, vy_3, x_3, y_3  = cv2.fitLine(np.array(C),cv2.DIST_L2,0,0.01,0.01)
    line = [float(vx_3),float(vy_3),float(x_3),float(y_3)]
    left_pt_3 = int((-x_3*vy_3/vx_3) + y_3)
    right_pt_3 = int(((img.shape[1]-x_3)*vy_3/vx_3)+y_3)
    #cv2.line(img,(img.shape[1]-1,right_pt_3),(0,left_pt_3),255,2)
    
            
    #vx_4, vy_4, x_4, y_4  = cv2.fitLine(np.array(D),cv2.DIST_L2,0,0.01,0.01)
    #line = [float(vx_4),float(vy_4),float(x_4),float(y_4)]
    #left_pt_4 = int((-x_4*vy_4/vx_4) + y_4)
    #right_pt_4 = int(((img.shape[1]-x_4)*vy_4/vx_4)+y_4)
    #cv2.line(img,(img.shape[1]-1,right_pt_4),(0,left_pt_4),255,2)
    return img
    
    
def get_color_goodfeatures(goodfeatures,img):
    goodfeatures = goodfeatures.reshape(-1,2)
    goodfeatures = np.int0(goodfeatures)
    
    color_goodfeatures = []
    for feature_coordinates in goodfeatures:
        feature_coordinates = feature_coordinates.reshape(2,-1)

        img_f = img[feature_coordinates[0,:],feature_coordinates[1,:]]
        color_goodfeatures.append(img_f)

    color_goodfeatures = np.asarray(color_goodfeatures)
    
    return color_goodfeatures


def get_sift(img, mask):
    mask = mask*255
    mask = np.uint8(mask)
    
    #img = np.asarray(img, np.float64)
    print(mask.shape)
    print(img.shape)

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(np.uint8(gray), np.uint8(gray), mask=mask)
    #cv2.imshow('sift', gray)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray,None)

    pts = cv2.KeyPoint_convert(keypoints)

    img=cv2.drawKeypoints(gray,keypoints,img)
    
    cv2.imshow('sift', img)

    return pts, descriptors
    

    
def left_parts_sift(pts, descriptors, img, connected_l):
    h, w = img.shape[:2]
    
    #print(pts)
    position = pts
    features = descriptors

    #print(features.shape)
    #print(position.shape)
    #print(len(features[0]))
    
    length_feature_x = len(features)
    length_feature_y = len(features[0]) + 2
    
    #print(length_feature_y)
    
    feature_set = np.empty((length_feature_x, length_feature_y))
    #print(feature_set)

    for i in range(length_feature_x):
        for j in range(length_feature_y):
            if(j == 0):
                feature_set[i,j] = position[i,0]
            if(j == 1):
                feature_set[i,j] = position[i,1]
            if(j > 1):
                feature_set[i,j] = features[i,j-2]

    #print(feature_set.shape)
    Z = np.float32(feature_set)
    #print(Z)
    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,3,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    
    A = Z[label.ravel()==0]
    B = Z[label.ravel()==1]
    C = Z[label.ravel()==2]
    
    

    print(position[0,1])
    print(Z[0,1])
    A = A[:,0:2]
    B = B[:,0:2]
    C = C[:,0:2]

    print(A.shape)
    A = np.uint8(A)
    for first_cluster in A:
        x,y = first_cluster.ravel()
        cv2.circle(img,(x,y),radius = 3,color=(0, 255, 0),thickness=-1)
        
    B = np.uint8(B)
    for second_cluster in B:
        x,y = second_cluster.ravel()
        cv2.circle(img,(x,y),radius = 3,color=(0, 0, 255),thickness=-1)
        
    C = np.uint8(C)
    for third_cluster in C:
        x,y = third_cluster.ravel()
        cv2.circle(img,(x,y),radius = 3,color=(255, 0, 0),thickness=-1)
        
  
    return img
    
    
    
    
    
    
    
    
    
    
    
