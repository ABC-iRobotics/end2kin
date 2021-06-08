from __future__ import print_function
from sklearn import preprocessing
import numpy as np
import cv2 as cv
from math import sqrt
import sys
import scipy.io
from io import StringIO
import optical_flow_utils as of



# author renataelek

if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    allFlow = of.optflow_main('/home/reni/IROB_projects/end2kin/virtualenvironment/JIGSAWS/Suturing/video/Suturing_B001_capture2.avi', 500, 640, 480)
    
    #optional file writing
    #writeToFile_flow(allFlow, 'allFlow_trial') 

    cv.destroyAllWindows()
