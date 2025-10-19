import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser(description="Object Detection on Video")
    parser.add_argument("--video", default="data/HDR80_A_Live_20231014_141501_000.mov", type=str, help="path to input video file")
    parser.add_argument("--model", default="./yolov8n_conti_1280_v1.pt", type=str, help="path to input video file")
    parser.add_argument("--device", default="0", type=str, help="device to run the model on")
    parser.add_argument("--write", action="store_true", help="write the output video to disk")
    args = parser.parse_args()
    # Load the YOLOv8 model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file {model_path} does not exist")
        return
    # Define the video path
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video file {video_path} does not exist")
        return
    print(f"Loading model from {model_path}")
    model = YOLO(str(model_path))
    device = int(args.device)
    model.to(device)
    
    print(f"Model loaded successfully, class names: {model.names}")
    # Open the video file
    video = cv2.VideoCapture(str(video_path))

    # Get the video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object to write the output video
    output_path = f"{video_path}_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if args.write:
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        output_video = None

    # Process each frame in the video and compute FPS
    sample_count = 0
    start_time = time.time()
    read_elapsed_time = 0
    while True:
        # Read the next frame
        read_st_time = time.time()
        ret, frame = video.read()
        # Break the loop if no more frames are available
        if not ret:
            break
        read_et_time = time.time()
        read_elapsed_time += read_et_time - read_st_time
        # print(f"Read frame delay {read_elapsed_time:.2f}")

        sample_count += 1
        if sample_count == 30:
            end_time = time.time()
            elapsed_time = end_time - start_time - read_elapsed_time
            fps = sample_count / elapsed_time
            print(f"Processed 30 frames in {elapsed_time:.2f} seconds, FPS: {fps:.2f}")
            sample_count = 0
            read_elapsed_time = 0
            start_time = time.time()

        # Perform object detection on the frame
        result = model.predict(frame, verbose=False, imgsz=1280, half=False)[0]
        boxes = result.boxes
        # Draw bounding boxes on the frame
        for xywh, class_id, conf in zip(boxes.xywh, boxes.cls, boxes.conf):
            x, y, w, h = xywh.tolist()
            label = model.names[class_id.item()]
            confidence = conf.item()
            # convert center xywh to xyxy
            x1, y1, x2, y2 = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        frame = result.plot()
        if output_video is not None:
            output_video.write(frame)

    # Release the video file and the output video
    video.release()
    if output_video is not None:
        output_video.release()

    print(f"Output video saved to {output_path}")

if __name__ == "__main__":
    main()