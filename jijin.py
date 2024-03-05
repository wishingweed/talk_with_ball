import cv2
import moviepy.editor as mp
from ultralytics import YOLO

def check_score(x1, y1, x2, y2):
    # (391, 245, 443, 295)
    if x1 > 371 and x2 < 463 and y1 < 265 and y2 > 275:
        return True
    else:
        return False

# 加载视频文件
video_file = "resource/0722VIDEO/score.MP4"
video_clip = mp.VideoFileClip(video_file)

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

# 初始化记录检测结果的变量
detected = False

yolo = YOLO('runs/detect/train4/weights/best.pt')

temp_score = 0
score = 0
rest = 0


# 遍历所有视频帧
for t in range(int(duration * fps)):
    # 获取当前时间点对应的帧图像
    frame = video_clip.get_frame(t / fps)
    results = yolo(frame)

    annotated_frame = results[0].plot()
    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    for ball_results in results:
        for bbox in ball_results.boxes.xyxy:
            # Get the coordinates of the bounding box.
            x1, y1, x2, y2 = [int(i) for i in bbox[:4]]
            # (391, 245, 443, 295)
            if check_score(x1, y1, x2, y2):
                print("score")
                temp_score += 1
                if rest > 10 and temp_score > 3:
                    score += 1
                    temp_score = 0
                    rest = 0
                    detected = True
                    start_t = max(0, t - 13 * fps)  # 记录视频起始时间
                    end_t = min(int(duration * fps), t + 2 * fps)
                    if detected and t >= end_t:
                        # 裁剪出对应的视频片段
                        sub_clip = video_clip.subclip(start_t / fps, end_t / fps)

                        # 添加片段到结果列表
                        result_frames.extend(list(sub_clip.iter_frames()))

                        # 重置标志位
                        detected = False
                        result_clip = mp.videotools.arrays_to_clip(result_frames, fps=fps)

                        # 保存结果视频
                        result_clip.write_videofile("jijin/result"+str(score)+".mp4", fps=fps, codec="libx264")

    rest += 1

if __name__ == '__main__':
    pass