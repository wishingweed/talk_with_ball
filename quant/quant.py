import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.quantization as quant

model = torch.load('../runs/detect/train4/weights/best.pt', map_location=torch.device('cpu'))

input_size = (1, 3, 224, 224)
input_data = torch.randn(input_size)

quantized_model = quant.quantize_dynamic(
    model, {nn.Conv2d}, dtype=torch.qint8
)

# 对量化模型进行评估
quantized_model.eval()

# 用输入数据进行测试
output = quantized_model(input_data)

# 输出量化模型的结构
print(quantized_model)

torch.save(quantized_model, 'yolo_v8_int8.pt')

if __name__ == '__main__':
    pass