import cv2
import moviepy.editor as mp
from ultralytics import YOLO


video_file = "resource/2024-01-28 071055.mov"
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
start_t = 12 * 60 + 40  # 12分40秒
end_t = 12 * 60 + 50  # 12分50秒

# sub_clip = video_clip.subclip(start_t, end_t)
cut = video_clip.subclip((12, 40), (12, 50))

# Save the result to a file
cut.write_videofile("cut_video.mov")

if __name__ == '__main__':
    pass