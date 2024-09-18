import cv2

# đọc video từ file

video = cv2.VideoCapture('D:/Python/opencv/pictures/3569286-uhd_3840_2160_24fps.mp4')

# tạo cửa sổ hiển thị
cv2.namedWindow("My video", cv2.WINDOW_NORMAL)


# hiển thị khung hình
while True:
    # đọc frame
    ret, frame = video.read()

    # exit khi không thấy frame
    if not ret :
        break

    # hiển thị
    cv2.imshow('Video player', frame)

    if(cv2.waitKey(10)==ord('q')):
        break

# Hủy bỏ player
video.release()
cv2.destroyAllWindows()