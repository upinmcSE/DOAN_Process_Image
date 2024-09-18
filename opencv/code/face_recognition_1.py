import cv2
import numpy as np

dir = 'D:/Python/opencv/pictures/face.png'

img = cv2.imread(dir)


# Tải mô hình đã được huấn luyện từ trước
net = cv2.dnn.readNetFromCaffe(
    'D:/Python/opencv/models/deploy.prototxt.txt',
    'D:/Python/opencv/models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
)

# Chuẩn bị dữ liệu đầu vào
# img: Đây là ảnh đầu vào mà bạn muốn nhận dạng khuôn mặt. 
#   Trong trường hợp của bạn, bạn đã đọc ảnh từ tệp "face.png".

# 1.0: Đây là tỷ lệ co giãn cho ảnh. Trong trường hợp này, 
#   ảnh sẽ không bị co giãn hoặc mở rộng, và giữ nguyên kích thước ban đầu.

# (300, 300): Đây là kích thước mà mô hình yêu cầu cho ảnh đầu vào.
#    Mô hình mà bạn đang sử dụng mong muốn ảnh có kích thước 300x300 pixel. 
#    Do đó, bạn co giãn hoặc cắt ảnh đầu vào thành kích thước này.

# (104, 177, 123): Đây là giá trị trung bình màu sắc được trừ đi từ mỗi pixel 
#   của ảnh. Điều này thường được sử dụng để chuẩn hóa dữ liệu đầu vào. 
#   Trong trường hợp này, các giá trị này thường được lấy từ dữ liệu huấn luyện của mô hình.

# swapRB=False: Đây là một cờ để xác định xem có cần hoán đổi các kênh màu đỏ và xanh (Red-Blue) 
#   trong ảnh hay không. Trong trường hợp này, bạn đã đặt nó thành False, tức là không hoán đổi kênh mà


# Đặt dữ liệu đầu vào cho mạng
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123), swapRB=False)

# Đặt dữ liệu đầu vào cho mạng
net.setInput(blob)

# Chạy mạng để phát hiện khuôn mặt
faces = net.forward()

# Lấy kích thước của ảnh đầu vào
h = img.shape[0]
w = img.shape[1]

# In thông tin
print(faces.shape)
print(faces[0, 0, 0, ])

# Duyệt từng khuôn mặt đã được phát hiện
for i in range(0, faces.shape[2]):
    confidence = faces[0,0,i,2]
    # Kiểm tra nếu mặt có độ tin cậy là 0.5
    if confidence>0.5:
        # Trích xuất tọa độ
        print(faces[0, 0, i, 3:7])
        startx = int(faces[0, 0, i, 3]*w)
        starty = int(faces[0, 0, i, 4]*h)
        endx = int(faces[0, 0, i, 5]*w)
        endy = int(faces[0, 0, i, 6]*h)
        print(startx, starty, endx, endy)
        
        # Vẽ hình chữ nhật xung quanh khuôn mặt đã phát hiện
        cv2.rectangle(img, (startx, starty), (endx, endy), (0, 255, 0),)
        
        # Hiển thị độ tin cậy
        text = 'Face: {:.2f}%'.format(confidence*100);
        cv2.putText(img, text, (startx, starty-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),)


cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()