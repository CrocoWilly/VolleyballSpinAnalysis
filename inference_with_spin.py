import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
import time
import numpy as np
import spin
import roiPreprocesser

def main():
    parser = argparse.ArgumentParser(description="Object Detection on Video")
    parser.add_argument("--video", default="./data/mikasa/HDR80_A_Live_20230211_153630_000.mov", type=str, help="path to input video file")
    parser.add_argument("--model", default="./yolov8n_mikasa_1280_v1.pt", type=str, help="path to input video file")
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
    
    model = YOLO(str(model_path))
    
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
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame in the video and compute FPS
    sample_count = 0
    start_time = time.time()
    read_elapsed_time = 0

    
    ## ------------NEW ADD-------------- ##
    # Ball Spin related things
    roiHandler = roiPreprocesser.RoiHandler()
    ballSpinCalculator = spin.BallSpinCalculator()
    frame_no = 0
    ## --------------------------------- ##

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


        ## ------------NEW ADD-------------- ##
        # Find the correct ball which are played in the game court, and exlcude the outside irrevalent balls
        draw_box = ballSpinCalculator.find_correct_ball(boxes)

        # Draw the correct ball's bounding box and calculate spin rate of it
        for index, (xywh, class_id, conf) in enumerate(zip(boxes.xywh, boxes.cls, boxes.conf)):
            
            x, y, w, h = xywh.tolist()
            label = model.names[class_id.item()]
            confidence = conf.item()

            ## Given the original size bounding box, then using roiHandler to preprocess it to 100*100
            if draw_box and (int(x) == draw_box[0] and int(y) == draw_box[1]):
                ## Define the target ball's roi
                roi = roiHandler.find_roi(frame, x, y, w, h)
                
                ## Preprocess the ball bounding box to 100*100 and stick it onto the left top of frame 
                upscaled_roi = roiHandler.enhance_image(roi)
                roiHandler.set_xyoffset(0, 0)
                frame[roiHandler.y_offset:roiHandler.y_offset+upscaled_roi.shape[0], roiHandler.x_offset:roiHandler.x_offset+upscaled_roi.shape[1]] = upscaled_roi

                ## Calculate spin rate: using Optical flow 
                frame, spin_rate = ballSpinCalculator.find_points_and_calculate_spin(frame, frame_no)

                ## Draw bounding box and spin rate
                x1, y1, x2, y2 = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"Spin: {spin_rate}rpm", (125, 75), cv2.FONT_HERSHEY_TRIPLEX, 2.5, (0, 0, 255), 5)

            # ## Show the frame with optical flow visualization
            # frameN = cv2.resize(frame, (1344, 756))
            # cv2.imshow('Results', frameN)
            # cv2.waitKey(200)


        output_video.write(frame)
        ballSpinCalculator.set_next_iteration()
        frame_no += 1
        ## --------------------------------- ##



    # Release the video file and the output video
    video.release()
    output_video.release()

    print(f"Output video saved to {output_path}")

if __name__ == "__main__":
    main()