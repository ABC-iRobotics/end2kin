import numpy as np
import cv2
import math
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot

def pose_estimation(im, point1, point2, point3, point4, point5, point6):
    # Read Image
    size = im.shape

    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                            point1.astype(np.float),            
                            point2,        
                            point3,     
                            point4,      
                            point5,  
                            point6     

                        ])
                    
    
    #print(image_points)


    # 3D model points.
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # axis
                            (-1.1, 0.3, 0.0),            # i dot
                            (-4.1, 0.3, 0.1),     # i bottom
                            (-5.5, -3.0, -4.0),      # front wheel
                            (-6.3, 0.0, 0.0),    # lower dot
                            (-8.1, -1.0, 0.0)      # lower corner

                        ])
                        
    # Camera intrinsic

    camera_matrix = np.array([[834.0711942257620, 0,379.9653989564771],[0, 851.6707768444244, 280.8233858394496],[0,0,1]],dtype = "double")
    
    dist_coeffs = np.array([-0.340002510589900,0.579575358194805 ,0,0])
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    (end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(im, (int(p[1]), int(p[0])), 3, (0,0,255), -1)

    p1 = ( int(image_points[0][1]), int(image_points[0][0]))

    p2 = ( int(end_point2D[0][0][1]), int(end_point2D[0][0][0]))

    cv2.line(im, p1, p2, (255,0,0), 2)

    # Display image
    cv2.imshow("Output", im)
    cv2.waitKey(25)
    return translation_vector


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        print("matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        print("matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
    
# function to display the coordinates of
# of the points clicked on the image

points = []
def click_event(event, y,x, flags, params):
    global points
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        #print(x, ' ', y)
        points.append([x,y])
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (y,x), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        #print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (y,x), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)
        
        
        
        
           
data = np.loadtxt('vid_tcp/vid_06_psm3_tcp.txt')


cap = cv2.VideoCapture('output_video6.avi')
image_coords_td = []
translation_vector = np.empty((3,0))
robot_pos = np.empty((3,0))

#150 fps
robot_frame_mul = 5 

i = 0

while(cap.isOpened()):
    ret, img = cap.read()
    
    if ret == True:
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', click_event)

        pressedKey = cv2.waitKey(0) & 0xFF
        
        if pressedKey == ord('q'):
            break
        elif not pressedKey == ord('n'):
            points = np.asarray(points)
            #print(points)
            translation_vector = np.append(translation_vector, pose_estimation(img, points[0,:], points[1,:], points[2,:], points[3,:], points[4,:], points[5,:]), axis = 1)

            data_pose_robot = np.array([data[i * robot_frame_mul,0:3]]).T * 1000.0      # m to mm
            
            robot_pos = np.append(robot_pos, data_pose_robot, axis = 1)
            #print(robot_pos)
               
            points = []
        i = i + 1
        print(i)
       
    else: 
        break
        
#print(translation_vector)
#print(robot_pos)

R, t = rigid_transform_3D(translation_vector, robot_pos)

print(R)
print(t)

#R, t = rigid_transform_3D(self.points_in_cam_base, points)

points_transformed = np.zeros(np.shape(translation_vector))
#print(translation_vector)
for i in range(np.shape(translation_vector)[1]):
    p = np.dot(R, translation_vector[:,i]) + t.T
    points_transformed[:,i] = p


fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(robot_pos[0,:], robot_pos[1,:], robot_pos[2,:], marker='o')
ax.scatter(points_transformed[0,:], points_transformed[1,:],
                                        points_transformed[2,:], marker='^')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
pyplot.show()

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()









