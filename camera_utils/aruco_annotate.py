import argparse
import cv2 as cv
from aruco.aruco_detect import ArucoDetectorHandler
from pathlib import Path
import json
import tqdm
from vidgear.gears import VideoGear


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
    parser.add_argument("--cam", type=str)
    parser.add_argument("--video", type=str)
    parser.add_argument("--mid", default=0, help="Main marker id")
    parser.add_argument("--othermid", default="1,2,3,4")
    parser.add_argument("--onlymid", action="store_true", help="Only save the main marker points")
    parser.add_argument("--ref", action="store_true", help="Save the points into refs")
    parser.add_argument("--refgap", default=15, type=int, help="Gap between frames")
    parser.add_argument("--out")
    args = parser.parse_args()
    camera_path = Path(args.cam)
    marker_id = args.mid
    # video_path = Path(args.video)
    input_path = args.video
    possible_exts = ["mov", "mp4"]
    cam_path = Path(args.cam)
    for ext in possible_exts:
        if input_path is None or not Path(input_path).exists():
            input_path = cam_path / f"{cam_path.name}.{ext}"
        if not Path(input_path).exists():
            input_path = cam_path.parent / f"{cam_path.name}.{ext}"
    if not Path(input_path).exists():
        raise FileNotFoundError(input_path)

    video_path = Path(input_path)
    cap = cv.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    out = None
    if args.out:
        out = cv.VideoWriter(args.out, cv.VideoWriter_fourcc(*'mp4v'), 
                             int(cap.get(cv.CAP_PROP_FPS) + .5), (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    frame_id = 0
    detector = ArucoDetectorHandler()
    fid_points = {}

    ref_fid_points = {}
    ref_other_fid_points = []
    other_mids = [int(mid) for mid in args.othermid.split(",")]
    tqdm_bar = tqdm.tqdm(total=total_frames, desc="Aruco Detection")
    cap = VideoGear(source=str(video_path), logging=False).start()
    while True:
        # ret, frame = cap.read()
        frame = cap.read()
        frame_id += 1
        # if not ret:
        if frame is None:
            break
        tqdm_bar.update(1)
        corners, ids, rejectedImgPoints = detector.detect_markers_in_image(frame)
        marker_dict = detector.corners_to_dict(corners, ids)

        if out:
            out_frame = detector.draw_markers(frame, corners, ids)
            out.write(out_frame)
        
        if not args.onlymid:
            for mid, corners in marker_dict.items():
                if mid == marker_id:
                    continue
                if mid not in other_mids:
                    continue
                ref_other_fid_points.append([pt.tolist() for pt in marker_dict[mid]])
                other_mids.remove(mid)

        if marker_id in marker_dict:
            fid_points[frame_id] = marker_dict[marker_id][0].tolist()
            if (frame_id - 1) % args.refgap == 0:
                ref_fid_points[frame_id] = [pt.tolist() for pt in marker_dict[marker_id]]
    if out:
        # out.release()
        out.close()
    
    common_dir = camera_path / "commons"
    common_dir.mkdir(exist_ok=True)
    with open(common_dir / f"aruco_{marker_id}.json", "w") as f:
        json.dump(fid_points, f)
    print(f"Other fids: {other_mids}, {other_mids}")
    if args.ref:
        ref_dir = camera_path / "refs"
        ref_dir.mkdir(exist_ok=True)
        ref_points = list(ref_fid_points.values()) + ref_other_fid_points
        with open(ref_dir / f"aruco_{marker_id}.json", "w") as f:
            json.dump([detector.aruco_unit_offset3d.tolist()] * len(ref_points), f)
        with open(ref_dir / f"aruco_{marker_id}_pts.json", "w") as f:
            json.dump(ref_points, f)
        
    
if __name__ == "__main__":
    main()