import numpy as np
import cv2
from matplotlib import pyplot as plt

# # Mouse callback function
def drawfunction(event,x,y,flags,param):
   if event == cv2.EVENT_LBUTTONDBLCLK:
      cv2.circle(img,(x,y),20,(255,255,255),-1)

dir = 'D:/Python/opencv/pictures/dog.jpg'
img = cv2.imread(dir)
cv2.namedWindow('image')
cv2.setMouseCallback('image',drawfunction)
while(1):
   cv2.imshow('image',img)
   key=cv2.waitKey(1)
   if key == ord('q'):
      break
cv2.destroyAllWindows()