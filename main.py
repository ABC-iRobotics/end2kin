# Import the necessary libraries
import cv2
import numpy as np
#import instrument_segmentation as segm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import joint_pose_module as jp
import pose_estimation as pe
import optical_flow_utils as of
import KalmanFilter  

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib
from scipy.spatial.transform import Rotation as Rot
import math 
from scipy.ndimage import gaussian_filter1d

# Define empty lists for measurements and predictions

meas=[]
pred=[]

meas_x=[]
pred_x=[]


##############################################################################
# Define various constants and parameters

diff_threshold = 0.3
p1_threshold = 40
p5_threshold = 40
weight_prev_pose = 0
weight_pose = 10
weight_prev_point = 0
weight_point = 10
prominence_joint1 = 14
filter_size = 25
cap = cv2.VideoCapture('output_video6_gt.avi')
data_robot = np.loadtxt('vid_tcp/vid_06_psm3_tcp.txt')
rgbcap = cv2.VideoCapture('output_video6.avi')
framerate = 7
translate_x = 15
translate_y = 15
translate_z = 10

##############################################################################
# Define a transformation matrix

R = [[ 0.92179532, -0.3872942 , -0.01722205],
 [ 0.08231492 , 0.23894111, -0.96753884],
 [ 0.37883723 , 0.89045514 , 0.25213488]]
t = [[-123.27196311 + translate_x],
 [  66.89844398 + translate_y],
 [ -76.23636566 + translate_z]]

R_reg = Rot.from_matrix(R)
R_euler = R_reg.as_euler('xyz', degrees=True)
R_euler_mod = [0,0,0]
R_euler_modified = R_euler + R_euler_mod
R_euler_modified = Rot.from_euler('xyz',R_euler_modified, degrees=True)
R = R_euler_modified.as_matrix()
print(R)

R = np.asarray(R)
t = np.asarray(t)


def paint(frame,meas,pred):
    for i in range(len(meas)-1): cv2.line(frame,meas[i],meas[i+1],(0,100,0))
    for i in range(len(pred)-1): cv2.line(frame,pred[i],pred[i+1],(0,0,200))
    
def meas_points(x,y):
    global mp,meas
    mp = np.array([[np.float32(x)],[np.float32(y)]])
    meas.append((x,y))
    
def meas_points_x(x,y):
    global mp_x,meas_x
    mp_x = np.array([[np.float32(x)],[np.float32(y)]])
    meas_x.append((x,y))
    
cv2.namedWindow("kalman")

# Define measurement and transition matrices for the Kalman Filters
kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 1


kalmanX = cv2.KalmanFilter(4,2)
kalmanX.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalmanX.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalmanX.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 1

mp = np.array((2,1), np.float32) # measurement
tp = np.zeros((2,1), np.float32) # tracked / prediction

mp_x = np.array((2,1), np.float32) # measurement
tp_x = np.zeros((2,1), np.float32) # tracked / prediction


############################################
#plt.ion()
#fig = plt.figure()
#ax = fig.gca(projection='3d')

#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
###########################################

# Initialize variables for robot position and error tracking
robot_frame_mul = framerate 
i = 0
robot_pos = np.empty((3,0))
#data_pose_robot = np.empty((3,1))

error_all_x = []
error_all_y = []
error_all_z = []

all_x = []
all_y = []
all_z = []
         
         
all_x_robot = []
all_y_robot = []
all_z_robot = []

joint_2_prev = []
point1p = np.array((193,427))

prevgray = of.get_grayImage(rgbcap)
    
# Loop through video frames        
#while(cap.isOpened()):
# Loop through video frames until two tools appear
for i in range(63):
    ret, frame = cap.read()
    print(i)
    if ret == True:

        closing, dist, thresh, img, img_g = jp.skeletonization(cap, filter_size)
        
        joint1, joint2, joint2_fromMean, thresh_show = jp.peak_calc(closing, dist, thresh, prominence_joint1)

        if(not joint2.any()):
            joint2 = joint2_prev
        if(joint1.any() and joint2.any()):
            print("peak exists")
            if(i == 0):
                prev_from_calc_0 = []
                prev_from_calc_upperB_0 = []
                prev_from_calc_lowerA_0 = []
                prev_from_calc_upperA_0 = []
                prev_from_calc_lastLowerB_0 = []
                prev_from_calc_lastUpperB_0 = []
                
                prev_iLowerB_0 = []
                prev_iUpperB_0 = []
                prev_lastiLowerB_0 = []
                prev_lastiUpperB_0 = []
                prev_iLower_0 = []
                prev_iUpper_0 = []
                
                
                point1, point2, point3, point4, point5, point6, prev_from_calc, prev_from_calc_upperB, prev_from_calc_lowerA, prev_from_calc_upperA, prev_from_calc_lastLowerB, prev_from_calc_lastUpperB, prev_iLowerB, prev_iUpperB, prev_iLower, prev_iUpper, prev_lastiLowerB, prev_lastiLowerB = jp.findNearestContour(closing, dist, joint2, joint1, cap,i, prev_from_calc_0, prev_from_calc_upperB_0, prev_from_calc_lowerA_0, prev_from_calc_upperA_0, prev_from_calc_lastLowerB_0, prev_from_calc_lastUpperB_0, prev_iLowerB_0, prev_iUpperB_0, prev_iLower_0, prev_iUpper_0, prev_lastiLowerB_0, prev_lastiUpperB_0)
                
            if(i > 0):

                point1_prev = point1
                point2_prev = point2
                point3_prev = point3
                point4_prev = point4
                point5_prev = point5
                point6_prev = point6
                
                point1, point2, point3, point4, point5, point6, prev_from_calc, prev_from_calc_upperB, prev_from_calc_lowerA, prev_from_calc_upperA, prev_from_calc_lastLowerB, prev_from_calc_lastUpperB,prev_iLowerB, prev_iUpperB, prev_iLower, prev_iUpper, prev_lastiLowerB, prev_lastiLowerB = jp.findNearestContour(closing, dist, joint2, joint1, cap,i, prev_from_calc, prev_from_calc_upperB, prev_from_calc_lowerA, prev_from_calc_upperA, prev_from_calc_lastLowerB, prev_from_calc_lastUpperB, prev_iLowerB, prev_iUpperB, prev_iLower, prev_iUpper, prev_lastiLowerB, prev_lastiLowerB)
                
                point5_curr = point5
                dxy5 = point5_curr - point5_prev
                point1_curr = point1
                dxy1 = point1_curr - point1_prev

                point1_curr = point1_prev + dxy5
                point2_curr = point2_prev + dxy5
                point3_curr = point3_prev + dxy5
                point4_curr = point4_prev + dxy1
                point5_curr = point5_prev + dxy1
                point6_curr = point6_prev + dxy1
                
                ###############################################################################################################################################
                if(abs(point1_curr[0] - point1[0]) > p1_threshold  or abs(point1_curr[1] - point1[1]) > p1_threshold or abs(point2_curr[0] - point2[0]) > p1_threshold  or abs(point2_curr[1] - point2[1]) > p1_threshold or abs(point3_curr[0] - point3[0]) > p1_threshold  or abs(point3_curr[1] - point3[1]) > p1_threshold):
                    print(point1_curr, point1)
                    point1 = ((point1_prev + dxy5)*weight_prev_point + point1*weight_point)/(weight_prev_point + weight_point)
                    point2 = ((point2_prev + dxy5)*weight_prev_point + point2*weight_point)/(weight_prev_point + weight_point)
                    point3 = ((point3_prev + dxy5)*weight_prev_point + point3*weight_point)/(weight_prev_point + weight_point)
                    print("diff is large")
            
                if(abs(point5_curr[0] - point5[0]) > p5_threshold  or abs(point5_curr[1] - point5[1]) > p5_threshold or abs(point6_curr[0] - point6[0]) > p5_threshold  or abs(point6_curr[1] - point6[1]) > p5_threshold or abs(point4_curr[0] - point4[0]) > p5_threshold  or abs(point4_curr[1] - point4[1]) > p5_threshold):
                    print(point1_curr, point1)
                    point4 = ((point4_prev + dxy1)*weight_prev_point + point4*weight_point)/(weight_prev_point + weight_point)
                    point5 = ((point5_prev + dxy1)*weight_prev_point + point5*weight_point)/(weight_prev_point + weight_point)
                    point6 = ((point6_prev + dxy1)*weight_prev_point + point6*weight_point)/(weight_prev_point + weight_point)
                    print("diff 5 is large")

                          
                ###############################################################################################################################################
        
            gray = of.get_grayImage(rgbcap)
        
            flow = of.optflow_main(prevgray, gray)
            
            flow_roi = flow[int(point4[0] - 50):int(point4[0]), int(point4[1]):int(point4[1] + 50), :]
            flow_roi_mean_x, flow_roi_mean_y = of.flow_mean(flow_roi)
            
            point6[1] = point6[1] + flow_roi_mean_y

            #2D image points. If you change the image, you need to change vector
            image_points = np.array([
                            point1.astype(np.float),            
                            point2,        
                            point3,     
                            point4,      
                            point5,  
                            point6     

                        ])
                              
        
        
            translation_vector, rotation_vector = pe.pose_estimation(thresh_show, point1, point2, point3, point4, point5, point6)        

            points_transformed = np.zeros(np.shape(translation_vector))
            for j in range(np.shape(translation_vector)[1]):
                p = np.dot(R, translation_vector[:,j]) + t.T
                points_transformed[:,j] = p
            
            
            rot_transformed = np.zeros(np.shape(rotation_vector))
            for k in range(np.shape(rotation_vector)[1]):
                rotv = np.dot(R, rotation_vector[:,k])
                rot_transformed[:,k] = rotv


            data_pose_robot = np.array([data_robot[i * robot_frame_mul,0:3]]).T * 1000.0      # m to mm
            
            robot_pos = np.append(robot_pos, data_pose_robot, axis = 1)
            data_ori_robot = np.array([data_robot[i * robot_frame_mul,3:12]])
            data_ori_robot = np.reshape(data_ori_robot, (3,3))
        
            data_ori_robot = Rot.from_matrix(data_ori_robot)
            data_ori_robot_euler = data_ori_robot.as_euler('zyx', degrees=True)
        
            ori_cam = np.reshape(rot_transformed, (3,))
            ori_cam = Rot.from_rotvec(ori_cam)
            ori_cam_euler = ori_cam.as_euler('zyx', degrees=True)
        
            i = i + 1
            
            points_transformed[0] = points_transformed[0] -2 
            points_transformed[1] = points_transformed[1]
            points_transformed[2] = points_transformed[2] + 3
            
            if(i == 1):
                prev_pose = points_transformed
            if(i > 1):
                prev_pose = prev_from_image
            if(abs(points_transformed[0] - prev_pose[0]) > diff_threshold  or abs(points_transformed[1] - prev_pose[1]) > diff_threshold  or abs(points_transformed[2] - prev_pose[2]) > diff_threshold ):
                points_transformed = (prev_pose*weight_prev_pose + points_transformed*weight_pose)/(weight_prev_pose + weight_pose)
                

            prev_from_image = points_transformed
            error_pos = data_pose_robot - points_transformed
            error_ori = data_ori_robot_euler - ori_cam_euler
        
            mp = np.array([[np.float32(points_transformed[1])],[np.float32(points_transformed[2])]])
            meas_points(points_transformed[1],points_transformed[2])
            kalman.correct(mp)
            tp = kalman.predict()
            pred.append((int(tp[0]),int(tp[1])))


            mp_x = np.array([[np.float32(points_transformed[0])],[np.float32(points_transformed[1])]])
            meas_points_x(points_transformed[0],points_transformed[1])

            kalmanX.correct(mp_x)
            tp_x = kalmanX.predict()
            pred_x.append((int(tp_x[0]),int(tp_x[1])))

            error_all_x.append(error_pos[0])
            error_all_y.append(error_pos[1])
            error_all_z.append(error_pos[2])
            
            all_x.append(points_transformed[0])
            all_y.append(points_transformed[1])
            all_z.append(points_transformed[2])
            
            all_x_robot.append(data_pose_robot[0])
            all_y_robot.append(data_pose_robot[1])
            all_z_robot.append(data_pose_robot[2])
            
            joint2_prev = joint2

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # Break the loop
    else:  
        break


pred = np.asarray(pred)
pred_x = np.asarray(pred_x)

error_all_x = np.array(error_all_x)
error_all_mean_x = np.mean(error_all_x)
error_all_y = np.array(error_all_y)
error_all_mean_y = np.mean(error_all_y)
error_all_z = np.array(error_all_z)
error_all_mean_z = np.mean(error_all_z)


all_x = np.array(all_x)
all_y = np.array(all_y)
all_z = np.array(all_z)

all_x = np.reshape(all_x, all_x.size)
all_x = jp.moving_average(all_x, 4)

all_y = np.reshape(all_y, all_y.size)
all_y = jp.moving_average(all_y, 4)

all_z = np.reshape(all_z, all_z.size)
all_z = jp.moving_average(all_z, 4)


all_x_robot = np.array(all_x_robot)
all_y_robot = np.array(all_y_robot)
all_z_robot = np.array(all_z_robot)

print(error_all_mean_x)
print(error_all_mean_y)
print(error_all_mean_z)

mse_kalman_x = np.mean(abs(all_x - all_x_robot))
mse_kalman_y = np.mean(abs(all_y - all_y_robot))
mse_kalman_z = np.mean(abs(all_z - all_z_robot))


print(mse_kalman_x)
print(mse_kalman_y)
print(mse_kalman_z)


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(all_x.flatten(), all_y.flatten(), all_z.flatten(), 'blue')

# Create a 3D plot for visualization
ax.plot3D(all_x_robot.flatten(), all_y_robot.flatten(), all_z_robot.flatten(), 'red')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_zlabel('z [mm]')
ax.set_title('video 6');
plt.show()

