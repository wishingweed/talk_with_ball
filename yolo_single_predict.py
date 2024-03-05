from PIL import Image
import cv2
from ultralytics import YOLO


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

def yolo_video_finetune_predict(model_path, video_path):
    yolo = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture('resource/0722VIDEO/score.MP4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = yolo(frame)

            # for ball_results in results:
            #     for bbox in ball_results.boxes.xyxy:
            #         # Get the coordinates of the bounding box.
            #         x1, y1, x2, y2 = [int(i) for i in bbox[:4]]
            #         # (391, 245, 443, 295)
            #         if check_score(x1, y1, x2, y2):
            #             temp_score += 1
            #             if rest > 10 and temp_score > 3:
            #                 current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            #                 time_list.append(current_time)
            #                 score += 1
            #                 temp_score = 0
            #                 rest = 0
            # rest += 1

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break



if __name__ == '__main__':
    # 'runs/detect/train7/weights/best.pt' gba 数据训练
    model_path = 'runs/detect/train7/weights/best.pt'
    image_path = 'resource/0128images/28.jpeg'
    # 高神video
    # video_path = 'resource/0722VIDEO/DJI_20230722104135_0025_D.MP4'
    # gba video
    video_path = 'resource/2024-01-28 071055.mov'

    # yolo_finetune_single_predict(model_path, image_path)
    yolo_video_finetune_predict(model_path, video_path)