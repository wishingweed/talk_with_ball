from ultralytics import YOLO

file_path = '/Users/wishingweed/Desktop/repos/basketball/dataset/0128_gbba_2.v3i.yolov8/data.yaml'

yolo = YOLO('yolov8n.pt')
yolo.train(data=file_path, epochs=200)
valid_results = yolo.val()
print(valid_results)

if __name__ == '__main__':
    pass