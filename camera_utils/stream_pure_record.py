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
def read_cap(cap, source_queue, fps=60):
    global is_terminated
    fps_s = 1 / fps
    while True:
        if is_terminated:
            break
        st_time = time.time()
        frame = cap.read()
        if frame is None:
            break
        source_queue.put(frame)
        ed_time = time.time()
        wait_time = fps_s - (ed_time - st_time)
        # if wait_time > 0:
        #     time.sleep(wait_time)
    source_queue.put(None)
    cap.close()

def find_nonexists_path(path):
    path = Path(path)
    i = 1
    orig_path = path
    while path.exists():
        path = orig_path.with_name(f"{orig_path.stem}_{i}{orig_path.suffix}")
        i += 1
    return str(path)

"""
    This tool is used to display the realtime video stream from the camera,
    and show the court lines computed by the camera calibrated parameters,
    so the user can align the camera manually because the camera's angle 
    have some error after each reboot.

    This is revised from non-pure version, 
    'pure' means it just record, I don't know why I designed the non-pure version that requires calibrated camset.
"""
def main():
    global is_terminated
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", help="Specify the camera sources")
    parser.add_argument("--names", nargs="*", help="Specify the camera names")
    parser.add_argument("--is_device", action="store_true")
    parser.add_argument("--display_size", type=str, default="1280,720", help="Specify the display size")
    parser.add_argument("--input_size", type=str, default="1920,1080", help="Specify the input size")
    parser.add_argument("--input_fps", type=int, default="60", help="Specify the input fps")
    parser.add_argument("--outdir", default="./record")
    parser.add_argument("--outfmt", default="{}.mp4")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file")
    parser.add_argument("--test", action="store_true", help="Test the camera, just display instead of record")
    args = parser.parse_args()
    sources = args.sources
    names = args.names
    outfmt = args.outfmt
    is_test = args.test
    if not names:
        names = [str(s) for s in sources]
    elif len(names) != len(sources):
        raise ValueError("names and sources should have the same length")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    if "," in args.display_size:
        display_size = tuple(map(int, args.display_size.split(",")))
    else:
        display_size = int(args.display_size)
    print("sources", sources)

    input_size = tuple(map(int, args.input_size.split(",")))
    input_fps = args.input_fps

    source_queues = {}
    source_threads = {}
    source_caps = {}
    source_outs = {}
    width, height = input_size
    options = {
        "CAP_PROP_FRAME_WIDTH": width,
        "CAP_PROP_FRAME_HEIGHT": height,
        "CAP_PROP_FPS": input_fps,
        'THREADED_QUEUE_MODE': True,
    }

    for i, source, name in zip(range(len(sources)), sources, names):
        if args.is_device:
            source = int(sources[i])
        else:
            source = sources[i]
        cap = VideoGear(source=source, logging=False, **options).start()
        source_caps[source] = cap
        camera_queue = Queue(maxsize=1)
        camera_thread = Thread(target=read_cap, args=(cap, camera_queue), daemon=True)
        source_queues[source] = camera_queue
        source_threads[source] = camera_thread
        # if isinstance(source, int):
        #     # out_path = outdir / f"{source}.mp4"
        #     out_path = outdir / args.outfmt.format(source)
        # else:
        #     # out_path = outdir / f"{source.name}.mp4"
        #     out_path = outdir / args.outfmt.format(source.name)
        out_path = outdir / outfmt.format(name)
        print(f"output path: {out_path}")
        if not args.overwrite:
            out_path = find_nonexists_path(out_path)
        # vcodec = "libopenh264"
        vcodec = "libx264"
        if is_test:
            out = None
        else:
            out = WriteGear(output=str(out_path), compression_mode=True, logging=False, \
                                    **{"-vcodec":vcodec, "-crf":0, "-preset":"slow", "-pix_fmt":"yuv420p", \
                                    "-input_framerate": input_fps, "-output_dimensions": (width, height)})
        source_outs[source] = out
    print("Press 'q' to quit")

    import signal
    import sys

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        for source, out in source_outs.items():
            out.close()
            print(f"{source} out closed")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to stop')

    for t in source_threads.values():
        t.start()
    while True:
        frames = []
        for source, camera_queue in source_queues.items():
            frame = camera_queue.get()
            source_outs[source].write(frame)
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

    is_terminated = True
    for source, out in source_outs.items():
        if out is not None:
            out.close()
        print(f"{source} out closed")

    for source, cap in source_caps.items():
        cap.stop()
        print(f"{source} cap stopped")
    cv.destroyAllWindows()
    print("out closed")

if __name__ == "__main__":
    main()