from ultralytics import YOLO

model = YOLO('yolov8n_conti_1280_v1.pt')
model.export(format='openvino')  # export to onnx