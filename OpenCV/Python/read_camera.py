#!/usr/bin/env python3

import sys
import cv2

# set Width and Height of output Screen 
frameWidth = 640
frameHeight = 480

# Use the first available camera
camera_id = 0
cap = cv2.VideoCapture(camera_id)
cap.set(3, frameWidth) 
cap.set(4, frameHeight) 

if (cap.isOpened()== False):
    print("Error opening video source")
    sys.exit(1)

# Read until video is stopped
while(cap.isOpened()):
    
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frame
        cv2.imshow('Frame', frame)
        
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()