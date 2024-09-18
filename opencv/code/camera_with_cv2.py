import cv2

camera = cv2.VideoCapture(0)

while(True):
    ret, frame = camera.read()

    if(not ret):
        break

    cv2.imshow('Resutl', frame)
    if(cv2.waitKey(1)==ord('q')):
        break

cv2.release()
cv2.destroyAllWindows()