from ultralytics import settings
from ultralytics import YOLO

def main():
    yolo_model_type = 'yolov8n'
    model = YOLO(f'{yolo_model_type}.pt')  # change the base model here
    # results = model.train(
    #     data='./datasets/ball_detection/conti_ball_detection.yaml',
    #     imgsz=640,
    #     epochs=300,
    #     batch=32,
    #     name='yolov8x_conti',
    #     device=[0, 1],
    #     resume=False,
    #     workers=16,
    #     lr0=0.0002,
    #     optimizer='Adam'
    # )

    # ball_name = "conti"
    # yaml_path = './datasets/ball_detection/conti_ball_detection.yaml'
    # image_size = 960
    image_size = (960, 540)
    ball_name = "mikasa"
    yaml_path = './datasets/ball_detection/mikasa_ball_detection.yaml'
    if type(image_size) == int:
        exp_name = f'{yolo_model_type}_{ball_name}_{image_size}'
    else:
        exp_name = f'{yolo_model_type}_{ball_name}_{image_size[0]}_{image_size[1]}'
    results = model.train(
        data=yaml_path,
        imgsz=image_size,
        epochs=300,
        batch=16,
        name=f'{yolo_model_type}_{ball_name}_{image_size}',
        device=[0, 1],
        resume=False,
        plots=True,
        profile=True,  # profile ONNX and TensorRT performance
        patience=30,
    )


if __name__ == '__main__':
    main()