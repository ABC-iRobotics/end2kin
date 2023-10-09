import cv2
import os

# Write synthetic MICCAI images to videos 
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video_writer = cv2.VideoWriter(filename='./output_video12.avi', fourcc=fourcc, fps=fps, frameSize=(701, 538))
for i in range(0,1000):
    p = i

    if os.path.exists('/home/reni/IROB_projects/end2kin/virtualenvironment/src/images_miccai/'+ '00' + str(p)+'.png'): #Judge whether the picture exists
        img = cv2.imread(filename='/home/reni/IROB_projects/end2kin/virtualenvironment/src/images_miccai/'+ '00' + str(p)+'.png')
        video_writer.write(img)
        print(str(p) + '.jpg' + ' done!')

    if os.path.exists('/home/reni/IROB_projects/end2kin/virtualenvironment/src/images_miccai/'+ '0' + str(p)+'.png'): #Judge whether the picture exists
        img = cv2.imread(filename='/home/reni/IROB_projects/end2kin/virtualenvironment/src/images_miccai/'+ '0' + str(p)+'.png')
        video_writer.write(img)
        print(str(p) + '.jpg' + ' done!')
    if os.path.exists('/home/reni/IROB_projects/end2kin/virtualenvironment/src/images_miccai/'+str(p)+'.png'): #Judge whether the picture exists
        img = cv2.imread(filename='/home/reni/IROB_projects/end2kin/virtualenvironment/src/images_miccai/'+ str(p)+'.png')
        video_writer.write(img)
        print(str(p) + '.jpg' + ' done!')
cv2.destroyAllWindows()
video_writer.release()


