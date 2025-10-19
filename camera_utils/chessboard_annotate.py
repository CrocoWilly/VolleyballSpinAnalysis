import cv2 as cv
import argparse
from pathlib import Path
import tqdm
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=str)
    parser.add_argument("--video", type=str)
    parser.add_argument("--chess", default="6x4")
    parser.add_argument("--out")
    parser.add_argument("--name", default="chess")
    parser.add_argument("--step", default=5, type=int)
    args = parser.parse_args()
    cam_path = Path(args.cam)
    name = args.name
    input_path = args.video
    possible_exts = ["mov", "mp4"]
    for ext in possible_exts:
        if input_path is None or not Path(input_path).exists():
            input_path = cam_path / f"{cam_path.name}.{ext}"
        if not Path(input_path).exists():
            input_path = cam_path.parent / f"{cam_path.name}.{ext}"
    if not Path(input_path).exists():
        raise FileNotFoundError(input_path)
    cap = cv.VideoCapture(str(input_path))
    fps = cap.get(cv.CAP_PROP_FPS)
    width, height = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Video {input_path} {width}x{height} {fps}fps")
    chess_size = tuple(map(int, args.chess.split("x")))
    objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)

    output_path = args.out
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
        if frame_id % args.step != 0:
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
    # print(points)
    print(f"Found {len(img_points)} chessboard corners")
    cap.release()
    if out is not None:
        out.release()
    (cam_path / "refs").mkdir(parents=True, exist_ok=True)
    with open(cam_path / "refs" / f"{name}.json", "w") as f:
        json.dump([pt.tolist() for pt in obj_points], f)
    with open(cam_path / "refs" / f"{name}_pts.json", "w") as f:
        json.dump([pt.tolist() for pt in img_points], f)

if __name__ == "__main__":
    main()

    
