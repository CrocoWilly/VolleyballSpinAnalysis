import argparse
import cv2 as cv
from aruco.aruco_detect import ArucoDetectorHandler
from pathlib import Path
import json
import tqdm
from vidgear.gears import VideoGear
from collections import defaultdict


def main():
    """
        Detect the aruco marker points, 
        sides of our aruco marker is sticked with same main point (top left corner), 
        so only the main point can be used to be common point for all cameras,
        the other points can only be used to calibrate the camera itself (intrinsic)

        This program mainly provide common points at first, but found out that good for calibration,
        so use the --ref option to save the points into refs for calibration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", type=str)
    parser.add_argument("--is_device", action="store_true")
    parser.add_argument("--mid", default=0, help="Main marker id")
    parser.add_argument("--out")
    args = parser.parse_args()
    sources = args.sources
    marker_id = args.mid
    if args.is_device:
        sources = [int(source) for source in sources]
    source_caps = {}
    total_width = 0
    total_height = 0
    width = 1920
    height = 1080
    input_fps = 60
    total_frames = 0
    options = {
        "CAP_PROP_FRAME_WIDTH": width,
        "CAP_PROP_FRAME_HEIGHT": height,
        "CAP_PROP_FPS": input_fps,
        'THREADED_QUEUE_MODE': True,
    }
    for source in sources:
        print(f"Opening Source: {source}")
        if not args.is_device and not Path(source).exists():
            raise ValueError(f"Source {source} not found")
        
        cap = VideoGear(source=str(source), logging=False, **options).start()
        cvcap = cv.VideoCapture(str(source))
        num_frames = cvcap.get(cv.CAP_PROP_FRAME_COUNT)
        total_width += width
        total_height = max(total_height, height)
        source_caps[source] = cap
        total_frames += num_frames

    out = None
    if args.out:
        out = cv.VideoWriter(args.out, cv.VideoWriter_fourcc(*'mp4v'), 
                             input_fps, (int(total_width), int(total_height)))

    frame_id = 0
    detector = ArucoDetectorHandler()
    fid_points = {}

    tqdm_bar = tqdm.tqdm(total=total_frames, desc="Aruco Detection")
    first_detected_frame_ids = {_:None for _ in sources}
    is_finished = False
    while not is_finished:
        frame_id += 1
        cap_out_frames = []
        for source, cap in source_caps.items():
            frame = cap.read()
            if frame is None:
                is_finished = True
                break
            tqdm_bar.update(1)
            # print(f"Frame: {frame}")
            corners, ids, rejectedImgPoints = detector.detect_markers_in_image(frame)

            marker_dict = detector.corners_to_dict(corners, ids)

            if out:
                out_frame = detector.draw_markers(frame, corners, ids)
                cap_out_frames.append(out_frame)

            if marker_id in marker_dict:
                fid_points[frame_id] = marker_dict[marker_id][0].tolist()
                if first_detected_frame_ids[source] is None:
                    first_detected_frame_ids[source] = frame_id
        if is_finished:
            break
        out_frame = cv.hconcat(cap_out_frames)
        if out:
            out.write(out_frame)
    if out:
        out.release()
    print()
    for source, first_frame_id in first_detected_frame_ids.items():
        print(f"Source: {source} First Aruco Frame Id: {first_frame_id}")
    
if __name__ == "__main__":
    main()