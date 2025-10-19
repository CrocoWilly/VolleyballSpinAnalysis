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
import time

is_terminated = False
def read_cap(cap, camera_queue, fps=60):
    fps_s = 1 / fps
    while True:
        if is_terminated:
            break
        st_time = time.time()
        frame = cap.read()
        if frame is None:
            break
        camera_queue.put(frame)
        ed_time = time.time()
        wait_time = fps_s - (ed_time - st_time)
        # if wait_time > 0:
        #     time.sleep(wait_time)
    camera_queue.put(None)
    cap.close()

def main():
    global is_terminated
    parser = argparse.ArgumentParser()
    parser.add_argument("--camset", type=str, help="Specify the camera set path, the camera directories is default to be under it", required=True)
    parser.add_argument("--cameras", nargs="+", help="Specify the camera names")
    parser.add_argument("--sources", nargs="+", help="Specify the camera sources")
    parser.add_argument("--is_device", action="store_true")
    parser.add_argument("--display_size", type=str, default="1280,720", help="Specify the display size")
    parser.add_argument("--input_size", type=str, default="1920,1080", help="Specify the input size")
    parser.add_argument("--input_fps", type=int, default="60", help="Specify the input fps")
    parser.add_argument("--outdir", default="./record")
    args = parser.parse_args()
    print("sources", args.sources)
    print("cameras", args.cameras)
    camset_path = Path(args.camset)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    camset_calib = CameraSetFileCalibrator(camset_path).load()
    camset = camset_calib.get_camset()
    if "," in args.display_size:
        display_size = tuple(map(int, args.display_size.split(",")))
    else:
        display_size = int(args.display_size)
    sources = args.sources
    input_size = tuple(map(int, args.input_size.split(",")))
    input_fps = args.input_fps
    if len(sources) != len(args.cameras):
        raise ValueError("The number of cameras and sources should be the same")
    cameras = []
    camera_name_camera_map = {camera.name: camera for camera in camset.cameras}
    for camera_name in args.cameras:
        if camera_name not in camera_name_camera_map:
            raise ValueError(f"Camera {camera_name} not found")
        camera = camera_name_camera_map[camera_name]
        cameras.append(camera)
    camset.cameras = cameras
    court_3d_pts = [[0, 0, 0], [9, 0, 0], [9, 18, 0], [0, 18, 0]]
    court_3d_pts = np.array(court_3d_pts, dtype=np.float32)
    camera_queues = {}
    camera_threads = {}
    camera_caps = {}
    camera_outs = {}
    width, height = input_size
    options = {
        "CAP_PROP_FRAME_WIDTH": width,
        "CAP_PROP_FRAME_HEIGHT": height,
        "CAP_PROP_FPS": input_fps,
        'THREADED_QUEUE_MODE': True,
    }
    camera_court_pts = {}
    for i, camera in enumerate(cameras):
        if args.is_device:
            source = int(sources[i])
        else:
            source = sources[i]
        cap = VideoGear(source=source, logging=False, **options).start()
        camera_caps[camera] = cap
        camera_queue = Queue(maxsize=1)
        camera_thread = Thread(target=read_cap, args=(cap, camera_queue))
        camera_queues[camera] = camera_queue
        camera_threads[camera] = camera_thread
        court_pts = cv.projectPoints(court_3d_pts, camera.rotation, camera.translation, camera.intrinsic, camera.distortion)[0]
        court_pts = court_pts.reshape(-1, 2)
        camera_court_pts[camera] = court_pts
        out_path = outdir / f"{camera.name}.mp4"
        vcodec = "libopenh264"
        vcodec = "libx264"
        out = WriteGear(output=str(out_path), compression_mode=True, logging=False, \
                                **{"-vcodec":vcodec, "-crf":0, "-preset":"slow", "-pix_fmt":"yuv420p", \
                                   "-input_framerate": input_fps, "-output_dimensions": (width, height)})
        camera_outs[camera] = out

    for t in camera_threads.values():
        t.start()
    while True:
        frames = []
        for camera, camera_queue in camera_queues.items():
            frame = camera_queue.get()
            camera_outs[camera].write(frame)
            if frame is None:
                break
            court_pts = camera_court_pts[camera]
            # draw as lines
            for i in range(4):
                cv.line(frame, [int(p) for p in court_pts[i]], \
                        [int(p) for p in court_pts[(i + 1) % 4]], (0, 0, 255), 2)
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
    is_terminated = True
    for camera, out in camera_outs.items():
        out.close()
    print("out closed")

if __name__ == "__main__":
    main()