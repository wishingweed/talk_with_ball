import cv2

# 读取resource/0722video 下所有视频path

image_index = 0
def get_video_path():
    import os
    folder_path = '/Users/wishingweed/Desktop/repos/basketball/resource/0722video'
    path_list = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(file_path)
            path_list.append(file_path)
    return path_list


# 读取视频


def video_image(path, image_folder, image_index, minutes, frame_interval):
    video = cv2.VideoCapture(path)

    # 设置间隔帧数


    # 保存为图片
    fps = video.get(cv2.CAP_PROP_FPS)

    # 从第十分钟开始
    frame_num = int(fps * 60 * minutes)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/output.mp4', fourcc, fps, size)



    while True:
        # 读取视频帧
        ret, frame = video.read()
        if video.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval == 0:
            out.write(frame)
            # frame 2 image
            try:
                cv2.imwrite(image_folder + str(image_index)+'.jpeg', frame)
            except:
                cv2.imwrite(image_folder + str(image_index)+'.png', frame)
            image_index += 1
            print(image_index)
            if image_index == 500:
                break
        if not ret:
            break
        # 按'q'键停止循环
        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    out.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()
    return image_index

# path_list = get_video_path()
# for path in path_list:
#     image_index = video_image(path, image_index)


if __name__ == '__main__':
    image_index = 0
    input_path = 'resource/cut_video.mp4'
    image_folder = './resource/0228images/'
    start_min = 0
    frame_interval = 2
    video_image(input_path, image_folder, image_index, start_min, frame_interval)
