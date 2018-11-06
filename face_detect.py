""" Experiment with face detection and image filtering using OpenCV """

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/richard/ImageProcessingToolbox/toolbox-computer-vision-rballaux/haarcascade_frontalface_alt.xml')
kernel = np.ones((40, 40), 'uint8')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    newFrame = cv2.flip(frame, 1) # flips the frame horizontally
    faces = face_cascade.detectMultiScale(newFrame, scaleFactor=1.2, minSize=(20, 20))
    for (x, y, w, h) in faces:
        #newFrame[y:y+h, x:x+w, :] = cv2.dilate(newFrame[y:y+h, x:x+w, :], kernel)
        cv2.rectangle(newFrame, (x, y), (x+w, y+h), (0, 0, 255))
        #drawing a mouth
        width_of_mouth = 5/10
        height_of_mouth = 1/8
        yposmouth = 5/6
        cv2.rectangle(newFrame, (int(x+w/2-width_of_mouth*w/2),int(y+yposmouth*h-height_of_mouth*h/2)),(int(x+w/2+width_of_mouth*w/2),int(y+yposmouth*h+height_of_mouth*h/2)),(255,0,0),-1)
        #drawing the eyes
        eyeradius = 1/13
        yposeye = 4/10
        xfromcenteroffseteye = 1/5
        cv2.circle(newFrame,(int(x+w/2-xfromcenteroffseteye*w),int(y+yposeye*h)),int(eyeradius*w),(255,0,0),-1)
        cv2.circle(newFrame,(int(x+w/2+xfromcenteroffseteye*w),int(y+yposeye*h)),int(eyeradius*w),(255,0,0),-1)
    # Display the resulting frame
    cv2.imshow('frame', newFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
