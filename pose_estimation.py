#!/usr/bin/env python

import cv2
import numpy as np

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


    # 3D model points (EndoWrist large needle driver)
    model_points = np.array([
                            (0.0,0.0,0.0),    # joint2 middle
                            (0.0, 2.4, 0.0),      # joint2 upper
                            (0.0,-2.4, 0.0),      # joint2 lower
                            (-10.0, 2.4, 0.0),             # joint1 upper
                            (-10.0, 0, 0.0),            # joint1 middle
                            (-10.0, -2.4, 0.0)    # joint1 lower

                        ])
                        

    # Camera intrinsic

    camera_matrix = np.array([[834.0711942257620, 0,379.9653989564771],[0, 851.6707768444244, 280.8233858394496],[0,0,1]],dtype = "double")

    #print{"Camera Matrix", camera_matrix}

    dist_coeffs = np.array([-0.340002510589900,0.579575358194805 ,0,0])
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(im, (int(p[1]), int(p[0])), 3, (0,0,255), -1)

    p1 = ( int(image_points[0][1]), int(image_points[0][0]))

    p2 = ( int(nose_end_point2D[0][0][1]), int(nose_end_point2D[0][0][0]))

    # Display image
    cv2.imshow("Output", im)
    cv2.waitKey(25)
    return translation_vector, rotation_vector

