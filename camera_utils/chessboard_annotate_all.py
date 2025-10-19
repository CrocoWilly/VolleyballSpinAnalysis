import cv2 as cv
import argparse
from pathlib import Path
import tqdm
import numpy as np
import json
import multiprocessing
import time
"""
    This script detect chessboard corners in a video and save the corners to a json file.
    the real length of the chessboard is not important because this is only used to calibrate the intrinsic parameters.
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=str)
    parser.add_argument("--chess", default="6x4", help="Specify the chessboard size in the format of NxM")
    parser.add_argument("--out", help="Specify the output directory")
    parser.add_argument("--step", default=30, type=int, help="Specify every N frames to detect the chessboard")

    args = parser.parse_args()
    cam_path = Path(args.cam)
    cam_name = cam_path.stem
    output_dir = args.out
    step = args.step
    chess_size = tuple(map(int, args.chess.split("x")))
    input_path_list = []
    output_path_list = []
    anno_name_list = []
    possible_exts = ["mov", "mp4"]
    print(f"Camera path: {cam_path}")
    for ext in possible_exts:
        # for path in [
        #     cam_path / f"{cam_path.name}.{ext}",
        #     cam_path.parent / f"{cam_path.name}.{ext}"
        # ]:
        pattern = f"chess_{cam_name}*.{ext}"
        print(f"Searching {pattern}")
        for path in list(Path(cam_path).rglob(pattern)):
            if not path.exists() or not path.is_file():
                continue
            print(f"Found video {path}")
            input_path_list.append(path)
            anno_name_list.append(path.stem)
            if output_dir is not None:
                output_path_list.append(output_dir / f"{path.stem}_cd.mp4")
            else:
                output_path_list.append(None)

    if len(input_path_list) == 0:
        raise Exception("No video found")
    
    with multiprocessing.Pool() as pool:
        task_args = [
            (input_path, cam_path, name, chess_size, step, output_path)
            for input_path, name, output_path in zip(input_path_list, anno_name_list, output_path_list)
        ]
        print("Start detecting chessboard")
        pool.starmap(detect_chessboard, task_args)
    print("Done")

def detect_chessboard(input_path, cam_path, anno_name, chess_size, step, output_path):
    cap = cv.VideoCapture(str(input_path))
    fps = cap.get(cv.CAP_PROP_FPS)
    width, height = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Video {input_path} {width}x{height} {fps}fps")
    objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)

    if output_path is not None:
        out = cv.VideoWriter(str(output_path), cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    else:
        out = None
    img_points = []
    obj_points = []
    tqdm_bar = tqdm.tqdm(total=int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
    frame_id = 0
    while True:
        frame_id += 1
        tqdm_bar.update(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % step != 0:
            continue
        found, corners = cv.findChessboardCorners(frame, chess_size)
        if found:
            corners = cv.cornerSubPix(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            corners = corners.reshape(-1, 2)
            img_points.append(corners)
            obj_points.append(objp)
            cv.drawChessboardCorners(frame, chess_size, corners, found)
        if out is not None:
            out.write(frame)
    print(f"Found {len(img_points)} chessboard corners")
    cap.release()
    if out is not None:
        out.release()
    (cam_path / "refs").mkdir(parents=True, exist_ok=True)
    with open(cam_path / "refs" / f"{anno_name}.json", "w") as f:
        json.dump([pt.tolist() for pt in obj_points], f)
    with open(cam_path / "refs" / f"{anno_name}_pts.json", "w") as f:
        json.dump([pt.tolist() for pt in img_points], f)

if __name__ == "__main__":
    main()

    
