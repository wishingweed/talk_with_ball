from PIL import Image
import cv2
from moviepy.editor import *
from ultralytics import YOLO


yolo = YOLO('runs/detect/train7/weights/best.pt')
filename = 'resource/0722VIDEO/DJI_20230722104135_0025_D.MP4'
tag = filename.split('.')[0].split('/')[-1]

cap = cv2.VideoCapture(filename)
# cap = cv2.VideoCapture('resource/0722VIDEO/score.MP4')

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = frame_count / fps

def check_score(x1, y1, x2, y2):
    # (391, 245, 443, 295)
    if x1 > 371 and x2 < 463 and y1 < 255 and y2 > 275:
        return True
    else:
        return False

def get_time_list():
    temp_score = 0
    score = 0
    rest = 0
    time_list = []
    # 设置间隔帧数
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = yolo(frame)

            for ball_results in results:
                for bbox in ball_results.boxes.xyxy:
                    # Get the coordinates of the bounding box.
                    x1, y1, x2, y2 = [int(i) for i in bbox[:4]]
                    # (391, 245, 443, 295)
                    if check_score(x1, y1, x2, y2):
                        temp_score += 1
                        if rest > 10 and temp_score > 3:
                            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                            time_list.append(current_time)
                            score += 1
                            temp_score = 0
                            rest = 0
            rest += 1

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
    return time_list



cap.set(cv2.CAP_PROP_POS_MSEC, 0)

time_list = get_time_list()

for i, time in enumerate(time_list):
    print(i, time)
    start_time = time - 8
    end_time = time + 2
    if start_time < 0:
        start_time = 0
    if end_time > duration:
        end_time = duration

    print(start_time, end_time)
    video_clip = VideoFileClip(filename).subclip(start_time, end_time)
    final_clip = concatenate_videoclips([video_clip])

    # 保存为MP4格式
    final_clip.write_videofile('jijin/output'+tag+str(i)+'.mp4', fps=fps, codec='libx264')



# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass