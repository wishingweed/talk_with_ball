import cv2
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch


video = cv2.VideoCapture('resource/sd1681176146_2.MP4')

# 设置间隔帧数
frame_interval = 2

# 加载对象检测器（例如Haar级联检测器或YOLO）
detector = ...


model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

fps = video.get(cv2.CAP_PROP_FPS)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/output.mp4', fourcc, fps, size)


def yolo_dector(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    image = Image.fromarray(image)
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    res = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # if model.config.id2label[label.item()] != 'person' and model.config.id2label[label.item()] != 'ball':
        #     continue
        res.append(box)
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )
    return res


while True:
    # 读取视频帧
    ret, frame = video.read()

    # 如果视频结束则停止循环
    if not ret:
        break

    # 每间隔帧数进行检测
    if video.get(cv2.CAP_PROP_POS_FRAMES) % frame_interval == 0:

        # 对帧进行检测
        detections = yolo_dector(frame)

        # 在帧上绘制圆圈
        for detection in detections:
            box = [int(i) for i in detection.tolist()]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

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