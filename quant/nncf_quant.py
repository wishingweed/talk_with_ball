from ultralytics import YOLO
from pathlib import Path
models_dir = Path('../models')
models_dir.mkdir(exist_ok=True)
DET_MODEL_NAME = "bbdint8"
det_model_path = models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"

print(det_model_path)

def export_openvino():
    det_model = YOLO('../runs/detect/train4/weights/best.pt')

    if not det_model_path.exists():
        det_model.export(format="openvino", dynamic=True, half=False)

from openvino.runtime import Core, Model

core = Core()
det_ov_model = core.read_model('../runs/detect/train4/weights/best_openvino_model/best.xml')
device = "CPU"  # "GPU"

det_compiled_model = core.compile_model(det_ov_model, device)






import nncf  # noqa: F811
from typing import Dict


def transform_fn(data_item:Dict):
    """
    Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
    Parameters:
       data_item: Dict with data item produced by DataLoader during iteration
    Returns:
        input_tensor: Input data for quantization
    """
    input_tensor = det_validator.preprocess(data_item)['img'].numpy()
    return input_tensor


quantization_dataset = nncf.Dataset(det_data_loader, transform_fn)


if __name__ == '__main__':
    pass