import cv2

# 打开视频文件
cap = cv2.VideoCapture('resource/sd1681176146_2.MP4')

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Failed to open video file.")
    exit()

# 循环读取视频帧
while True:
    # 读取视频帧
    ret, frame = cap.read()

    # 如果读取到最后一帧，退出循环
    if not ret:
        break

    # 在这里添加对视频帧的处理代码
    # ...

    # 在这里进行渲染并显示视频帧
    cv2.imshow('frame', frame)

    # 等待下一帧
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭视频文件和窗口
cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass