from PIL import Image
import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import defaultdict
import moviepy.editor as mp
from collections import deque

# 定义一个最大长度为5的队列
side_queue = deque(maxlen=5)



distance_thresh = 500


class Point:
    def __init__(self, xy):
        self.x = xy[0]
        self.y = xy[1]


def line_intersection(line1, line2):
    """
    :param line1: 线段一开头结尾坐标[[x1,y1], [x2,y2]]
    :param line2: 线段二开头结尾坐标[[x3,y3], [x4,y4]]
    :return: False/[x,y]
    """
    a, b, c, d = Point(line1[0]), Point(line1[1]), Point(line2[0]), Point(line1[1])
    # 三角形abc 面积的2倍
    area_abc = (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
    # 三角形abd 面积的2倍
    area_abd = (a.x - d.x) * (b.y - d.y) - (a.y - d.y) * (b.x - d.x)
    # 面积符号相同则两点在线段同侧, 不相交(对点在线段上的情况, 本例当作不相交处理);
    if area_abc * area_abd >= 0:
        return False
    # 三角形cda面积的2倍
    area_cda = (c.x - a.x) * (d.y - a.y) - (c.y - a.y) * (d.x - a.x)
    # 三角形cdb面积的2倍
    # 注意: 这里有一个小优化.不需要再用公式计算面积, 而是通过已知的三个面积加减得出.
    area_cdb = area_cda + area_abc - area_abd
    if area_cda * area_cdb >= 0:
        return False
    # 计算交点坐标
    t = area_cda / (area_abd - area_abc)
    dx = t * (b.x - a.x)
    dy = t * (b.y - a.y)
    return [a.x + dx, a.y + dy]


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
    basket_positions = []

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
            basket_results = []

            basket_x1 = 0
            basket_y1 = 0
            basket_x2 = 0
            basket_y2 = 0

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
                    center_x = int((x1+x2)/2)
                    basket_results.append((center_x, y1, x1, y1, x2, y2))

                if result_class == 1:
                    x1, y1, x2, y2 = [int(i) for i in results[0].boxes.xyxy[i]]
                    center_x = int((x1+x2)/2)
                    center_y = int((y1+y2)/2)
                    ball_results.append((center_x, center_y))


            cur_basket_position = (0, 0)
            for basket_result in basket_results:
                cur_distance_x = abs(basket_result[0] -video_center)
                if cur_distance_x < 200:
                    cur_basket_position = basket_result
                if basket_result[0] -video_center > 0:
                    side_queue.append('right')
                else:
                    side_queue.append('left')

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
            if len(ball_positions) > 2 and cur_basket_position != (0, 0):
                joint_pos = line_intersection([[
                    , cur_ball_y], [ball_positions[-2][0], ball_positions[-2][1]]], [[cur_basket_position[2], cur_basket_position[3]],[cur_basket_position[4], cur_basket_position[5]]])
                print(joint_pos)
                print([[cur_ball_x, cur_ball_y], [ball_positions[-2][0], ball_positions[-2][1]]])
                print([[cur_basket_position[2], cur_basket_position[3]],[cur_basket_position[4], cur_basket_position[5]]])
                if joint_pos:
                    print('进了么')


            # all_left = all(item == 'left' for item in queue)





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