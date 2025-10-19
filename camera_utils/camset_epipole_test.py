import argparse
from .calibration import CameraFileCalibrator, CameraSetFileCalibrator
from .camera import Camera, CameraSet
import cv2 as cv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def draw_line(img, line, color, thickness=3, line_type=cv.LINE_AA):
    h, w, _ = img.shape
    a, b, c = line
    if abs(a) > abs(b):
        y0 = 0
        y1 = h
        x0 = int((-c - b*y0) / a)
        x1 = int((-c - b*y1) / a)
    else:
        x0 = 0
        x1 = w
        y0 = int((-c - a*x0) / b)
        y1 = int((-c - a*x1) / b)
    img = cv.line(img, (x0, y0), (x1, y1), color, thickness, line_type)
    return img


parser = argparse.ArgumentParser(description="Camera Set Epipole Test")
parser.add_argument("camset_dir", type=str, help="path to the camera set directory")
args = parser.parse_args()

camset_dir = Path(args.camset_dir)
if not camset_dir.exists():
    print(f"Camera set directory {camset_dir} does not exist")
    exit()

calibrator = CameraSetFileCalibrator(camset_dir).load()
camset = calibrator.get_camset()
cameras = camset.cameras
sample_pt3ds = [[0,0,0], [0,6,0], [0,9,0], [0,12,0], [0,18,0], [9,0,0], [9,6,0], [9,9,0], [9,12,0], [9,18,0]]
color_map = plt.cm.get_cmap("rainbow")
for camera in cameras:
    camera: Camera
    img_path = camset_dir / camera.name / "samples" / "1.jpg"
    out_img_path = camset_dir / camera.name / "etest.jpg"
    if not img_path.exists():
        print(f"Image file {img_path} does not exist")
        continue
    img = cv.imread(str(img_path))
    if img is None:
        print(f"Failed to read image file {img_path}")
        continue
    pt2ds, _ = cv.projectPoints(np.array(sample_pt3ds, dtype=np.float32), camera.rotation, camera.translation, camera.intrinsic, camera.distortion)
    pt2ds = pt2ds.reshape(-1, 2).astype(np.int32)
    
    for other_idx, other_cam in enumerate(cameras):
        if other_cam == camera:
            continue
        cam_color = color_map(other_idx / len(cameras))
        other_cam: Camera
        F = camset.get_fmat(camera, other_cam)
        other_pt2ds, _ = cv.projectPoints(np.array(sample_pt3ds, dtype=np.float32), other_cam.rotation, other_cam.translation, other_cam.intrinsic, other_cam.distortion)
        other_pt2ds = other_pt2ds.reshape(-1, 2).astype(np.int32)
        for pt2d in other_pt2ds:
            line = camset.get_epiline(camera, other_cam, pt2d, F, 2)
            print(f"Epiline: {line}")
            draw_line(img, line, [int(c*255) for c in cam_color[:3]], thickness=2)
    for pt3d, pt2d in zip(sample_pt3ds, pt2ds):
        print(f"3D: {pt3d}, 2D: {pt2d}")
        cv.circle(img, pt2d, 5, (100, 0, 100), -1) 

    cv.imwrite(str(out_img_path), img)
