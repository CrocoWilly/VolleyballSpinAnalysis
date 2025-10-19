from .calibration import CameraFileCalibrator, CameraSetFileCalibrator
import cv2 as cv
from pathlib import Path
import argparse
import numpy as np
import itertools
import matplotlib.pyplot as plt
import tqdm
from vidgear.gears import WriteGear, VideoGear
import json

def draw_line(img, line, color, thickness=3):
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [img.shape[1], -(line[2] + line[0] * img.shape[1]) / line[1]])
    img = cv.line(img, (x0, y0), (x1, y1), color, thickness)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camset", type=str, help="Specify the camera set path, the camera directories is default to be under it", required=True)
    parser.add_argument("--outname", type=str, default="epiline")
    parser.add_argument("--common", default=None, help="The common file name, set to None means from the camera set file")
    parser.add_argument("--nocourt", action="store_true", help="Draw the court")
    parser.add_argument("--pair", default=None, help="Specify the pair of cameras to compare, if not specified, compare all")
    args = parser.parse_args()
    camset_path = Path(args.camset)
    camset_calib = CameraSetFileCalibrator(camset_path)
    camset_calib.load()
    camset = camset_calib.get_camset()
    cmap = plt.get_cmap("plasma")
    common_name = args.common

    for cam1, cam2 in itertools.combinations(camset.cameras, 2):
        if f"{cam1.name},{cam2.name}" != args.pair and f"{cam2.name},{cam1.name}" != args.pair and args.pair is not None:
            continue
        video_path1 = camset_calib.get_sample_video_path(cam1.name)
        video_path2 = camset_calib.get_sample_video_path(cam2.name)
        if video_path1 is None or not video_path1.exists():
            continue
        print(video_path1)
        cap1 = cv.VideoCapture(str(video_path1))
        cap2 = cv.VideoCapture(str(video_path2))
        cap_w, cap_h = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
        cap_fps = cap1.get(cv.CAP_PROP_FPS)

        total_frames = min(int(cap1.get(cv.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv.CAP_PROP_FRAME_COUNT)))
        out_path = str(camset_path / f"{cam1.name}_{cam2.name}_{args.outname}.mp4")
        # out = cv.VideoWriter(out_path, cv.VideoWriter_fourcc(*'mp4v'), cap_fps, (cap_w * 2, cap_h))
        out = WriteGear(output=out_path, compression_mode=True, logging=False, \
                        **{"-vcodec":"libx264", "-crf":22, "-preset":"fast",
                           "-input_framerate": 60, "-output_dimensions": (cap_w * 2, cap_h)})
        cam1_points = {}
        cam2_points = {}
        if common_name is not None:
            cam1_common_path = camset_path / cam1.name/ "commons" / common_name
            cam2_common_path = camset_path / cam2.name/ "commons" / common_name
            if not cam1_common_path.exists():
                raise FileNotFoundError(cam1_common_path)
            if not cam2_common_path.exists():
                raise FileNotFoundError(cam2_common_path)
            with open(cam1_common_path, "r") as f:
                cam1_common = json.load(f)
                cam1_common = {int(k):v for k,v in cam1_common.items()}
            with open(cam2_common_path, "r") as f:
                cam2_common = json.load(f)
                cam2_common = {int(k):v for k,v in cam2_common.items()}
            print("cam1_common", len(cam1_common))
            print("cam2_common", len(cam2_common))
            cam1_common = {common_name: cam1_common}
            cam2_common = {common_name: cam2_common}
        else:
            cam1_common = cam1.commons
            cam2_common = cam2.commons

        for common_type, common_fid_pts in cam1_common.items():
            for fid, pts in common_fid_pts.items():
                if fid not in cam1_points:
                    cam1_points[fid] = []
                cam1_points[fid].append(pts)
        for common_type, common_fid_pts in cam2_common.items():
            for fid, pts in common_fid_pts.items():
                if fid not in cam2_points:
                    cam2_points[fid] = []
                cam2_points[fid].append(pts)
        fmat = camset.get_fmat(cam1, cam2)
        frame_id = 0
        tqdm_bar = tqdm.tqdm(total=total_frames)
        cap1 = VideoGear(source=str(video_path1), logging=False).start()
        cap2 = VideoGear(source=str(video_path2), logging=False).start()
        court_3d_pts = np.array([[0, 0, 0], [9, 0, 0], [9, 18, 0], [0, 18, 0]], dtype=np.float32)
        cam1_court_pts = cv.projectPoints(court_3d_pts, cam1.rotation, cam1.translation, cam1.intrinsic, cam1.distortion)[0]
        cam2_court_pts = cv.projectPoints(court_3d_pts, cam2.rotation, cam2.translation, cam2.intrinsic, cam2.distortion)[0]
        cam1_court_pts = cam1_court_pts.reshape(-1, 2)
        cam2_court_pts = cam2_court_pts.reshape(-1, 2)
        while True:
            # ret1, frame1 = cap1.read()
            # ret2, frame2 = cap2.read()
            frame1 = cap1.read()
            frame2 = cap2.read()
            frame_id += 1
            tqdm_bar.update(1)
            # if not ret1 or not ret2:
            #     break
            if frame1 is None or frame2 is None:
                break
            if not args.nocourt:
                for i in range(len(court_3d_pts)):
                    cv.line(frame1, tuple(cam1_court_pts[i-1].astype(int)), tuple(cam1_court_pts[i].astype(int)), (0, 0, 255), 2)
                    cv.line(frame2, tuple(cam2_court_pts[i-1].astype(int)), tuple(cam2_court_pts[i].astype(int)), (0, 0, 255), 2)
            if frame_id in cam1_points:
                pts1 = np.array(cam1_points[frame_id])
                lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fmat)
                lines2 = lines2.reshape(-1, 3)
                idx = 0
                for r, pt in zip(lines2, pts1):
                    idx = (idx + 1) % len(lines2)
                    color = cmap(idx / len(lines2))
                    color = [int(c*255) for c in color]
                    frame2 = draw_line(frame2, r, color)
                    frame1 = cv.circle(frame1, (pt).astype(int), 5, color, -1)
            if frame_id in cam2_points:
                pts2 = np.array(cam2_points[frame_id])
                lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fmat)
                lines1 = lines1.reshape(-1, 3)
                idx = 0
                for r, pt in zip(lines1, pts2):
                    idx = (idx + 1) % 20
                    color = cmap(idx / 20)
                    color = [int(c*255) for c in color]
                    frame1 = draw_line(frame1, r, color)
                    frame2 = cv.circle(frame2, (pt).astype(int), 5, color, -1)
                
            tqdm_bar.set_description(f"cam {cam1.name} {cam2.name} pts1={len(pts1) if frame_id in cam1_points else 0}, pts2={len(pts2) if frame_id in cam2_points else 0}")
            frame = np.concatenate((frame1, frame2), axis=1)
            out.write(frame)
        out.close()

if __name__ == "__main__":
    main()