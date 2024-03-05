from PIL import Image
import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import defaultdict
import moviepy.editor as mp

distance_thresh = 500


def origin_predict():
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
    res = model('resource/0722images/1.jpeg', conf=0.1)  # return a list of Results objects
    res_plotted = res[0].plot()
    cv2.imshow("result", res_plotted)
    cv2.waitKey()
    # Process results list

def yolo_finetune_single_predict(model_path, image_path):
    yolo = YOLO(model_path)
    res = yolo(image_path, conf=0.2)  # return a list of Results objects
    res_plotted = res[0].plot()
    cv2.imshow("result", res_plotted)
    cv2.waitKey()

def get_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_ball_distance(x1, y1, positions):
    if len(positions) > 0:
        return get_distance(x1, y1, positions[-1][0], positions[-1][1])
    else:
        return 0

def is_score():
    pass

def yolo_video_finetune_predict(model_path, video_path):
    yolo = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    video_clip = mp.VideoFileClip(video_path)

    # 视频帧率
    fps = video_clip.fps
    # 视频总时长
    duration = video_clip.duration
    # 视频宽度
    width = video_clip.w

    # 视频高度
    height = video_clip.h

    # 初始化结果视频
    result_clip = mp.VideoClip()
    result_frames = []
    int_name_dict = {0: 'backboard', 1: 'ball', 2: 'basket'}
    ball_positions = []

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        video_center = frame.shape[1]/2
        video_width = frame.shape[1]
        track_history = defaultdict(lambda: [])
        if success:
            # Run YOLOv8 inference on the frame
            results = yolo(frame, conf=0.2)

            annotated_frame = results[0].plot()
            # int_name_dict = {0: 'backboard', 1: 'ball', 2: 'basket'}
            ball_results = []

            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                result_class = int(box.cls)
                try:
                    score = float(box.conf)
                except Exception as e:
                    score = 0
                if result_class == 0:
                    x1, y1, x2, y2 = [int(i) for i in results[0].boxes.xyxy[i]]
                    center_obj_x = (x1 + x2)/2
                    if center_obj_x > video_width*2/3:
                        print('right side of court')
                        cbackboard_x1, cbackboard_y1, cbackboard_x2, cbackboard_y2 = x1, y1, x2, y2

                if result_class == 2:
                    x1, y1, x2, y2 = [int(i) for i in results[0].boxes.xyxy[i]]

                if result_class == 1:
                    x1, y1, x2, y2 = [int(i) for i in results[0].boxes.xyxy[i]]
                    center_x = int((x1+x2)/2)
                    center_y = int((y1+y2)/2)
                    ball_results.append((center_x, center_y))

            min_distance = 1000
            cur_ball_x = 0
            cur_ball_y = 0
            for ball_result in ball_results:
                cur_distance = get_ball_distance(ball_result[0], ball_result[1], ball_positions)
                if cur_distance < min_distance:
                    cur_ball_x = ball_result[0]
                    cur_ball_y = ball_result[1]
                    min_distance = cur_distance
            if min_distance < distance_thresh:
                ball_positions.append((cur_ball_x, cur_ball_y))



                # if len(ball_positions) > 0:
                #     last_distance = get_distance(center_x, center_y, ball_positions[-2][0], ball_positions[-2][1])
                #
                # else:
                #     ball_positions.append((center_x, center_y))

            for position in ball_positions:
                cv2.circle(frame, position, 3, (0, 255, 0), -1)
            # cv2.imshow('frame', frame)

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break




if __name__ == '__main__':
    # 'runs/detect/train7/weights/best.pt' gba 数据训练
    model_path = 'runs/detect/train8/weights/best.pt'
    image_path = 'resource/0128images/28.jpeg'
    # 高神video
    # video_path = 'resource/0722VIDEO/DJI_20230722104135_0025_D.MP4'
    # gba video
    video_path = 'resource/cut_video.mp4'

    # yolo_finetune_single_predict(model_path, image_path)
    yolo_video_finetune_predict(model_path, video_path)