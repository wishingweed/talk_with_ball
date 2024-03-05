import cv2
import numpy as np
import time
video = cv2.VideoCapture('resource/sd1681176146_2.MP4')
from ultralytics import YOLO

# 设置间隔帧数
frame_interval = 2


# Load the YOLO models
ball_model = YOLO("basketballModel.pt")

fps = video.get(cv2.CAP_PROP_FPS)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.mp4', fourcc, fps, size)



while True:
    # 读取视频帧
    ret, frame = video.read()

    # 如果视频结束则停止循环
    if not ret:
        break

    # 每间隔帧数进行检测
    if video.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval == 0:

        # 对帧进行检测
        ball_results_list = ball_model(frame, verbose=False, conf=0.1)

        for ball_results in ball_results_list:
            for bbox in ball_results.boxes.xyxy:
                # Get the coordinates of the bounding box.
                x1, y1, x2, y2 = [int(i) for i in bbox[:4]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 在帧上绘制圆圈
        # for detection in detections:
        #     box = [int(i) for i in detection.tolist()]
        #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # 显示帧
    cv2.imshow('frame', frame)

    # 按'q'键停止循环
    if cv2.waitKey(1) == ord('q'):
        break
    out.write(frame)

# 释放视频
video.release()
out.release()
# 关闭所有窗口
cv2.destroyAllWindows()



if __name__ == '__main__':
    pass