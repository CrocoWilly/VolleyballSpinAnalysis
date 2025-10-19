"""
    This tool is used to display the realtime video stream from the camera,
    and show the court lines computed by the camera calibrated parameters,
    so the user can align the camera manually because the camera's angle 
    have some error after each reboot.
"""


import cv2 as cv
from vidgear.gears import VideoGear, WriteGear
from pathlib import Path
import argparse
from .calibrate_tool import CameraFileCalibrator, CameraSetFileCalibrator
from .camera import Camera, CameraSet
import numpy as np
from threading import Thread
from queue import Queue
import imutils

def read_cap(cap, camera_queue):
    while True:
        frame = cap.read()
        if frame is None:
            break
        camera_queue.put(frame)
    camera_queue.put(None)
    cap.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", help="Specify the camera sources")
    parser.add_argument("--is_device", action="store_true")
    parser.add_argument("--display_size", type=str, default="1280,720", help="Specify the display size")
    parser.add_argument("--input_size", type=str, default="1920,1080", help="Specify the input size")
    parser.add_argument("--input_fps", type=int, default="60", help="Specify the input fps")
    args = parser.parse_args()
    print("sources", args.sources)
    if "," in args.display_size:
        display_size = tuple(map(int, args.display_size.split(",")))
    else:
        display_size = int(args.display_size)
    sources = args.sources
    input_size = tuple(map(int, args.input_size.split(",")))
    input_fps = args.input_fps
    cameras = []
    camera_queues = {}
    camera_threads = {}
    camera_caps = {}
    width, height = input_size
    options = {
        "CAP_PROP_FRAME_WIDTH": width,
        "CAP_PROP_FRAME_HEIGHT": height,
        "CAP_PROP_FPS": input_fps,
        'THREADED_QUEUE_MODE': True,
    }
    camera_court_pts = {}
    for cam_id in range(len(sources)):
        if args.is_device:
            source = int(sources[cam_id])
        else:
            source = sources[cam_id]
        cap = VideoGear(source=source, logging=False, **options).start()
        camera_caps[cam_id] = cap
        camera_queue = Queue(maxsize=1)
        camera_thread = Thread(target=read_cap, args=(cap, camera_queue))
        camera_queues[cam_id] = camera_queue
        camera_threads[cam_id] = camera_thread

    for t in camera_threads.values():
        t.start()
    frame_id = 0
    while True:
        frame_id += 1
        frames = []
        for cam_id, camera_queue in camera_queues.items():
            frame = camera_queue.get()
            if frame is None:
                break
            frames.append(frame)
        if len(frames) == 0:
            break
        frame = np.hstack(frames)
        if isinstance(display_size, int):
            frame = imutils.resize(frame, width=display_size)
        else:
            frame = imutils.resize(frame, width=display_size[0], height=display_size[1])
        cv.imshow("cameras", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break

if __name__ == "__main__":
    main()