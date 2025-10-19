import argparse
from .calibration import CameraFileCalibrator, CameraSetFileCalibrator
import cv2 as cv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=str)
    parser.add_argument("--base", type=str, help="Specify the base camera path")
    parser.add_argument("--camset", type=str, help="Specify the camera set path, the camera directories is default to be under it")
    parser.add_argument("--cams", type=str)
    parser.add_argument("--camdir", type=str, help="Override the directory of the cameras")
    parser.add_argument("--common", action="store_true", help="Use common points to compute fmat")
    parser.add_argument("--infix", action="store_true", help="Fix intrinsic parameters")
    parser.add_argument("--inguess", action="store_true", help="Use intrinsic guess")
    parser.add_argument("--show", action="store_true", help="Show the calibration result")
    args = parser.parse_args()
    
    if args.cam is None and args.camset is None:
        raise ValueError("Either cam or camset should be specified")
    if args.camset is None:
        if args.cam is None:
            raise ValueError("cam should be specified")
        calibrator = CameraFileCalibrator(args.cam)
        if args.show:
            print("Showing calibration result")
            calibrator.load()
            camera = calibrator.get_camera()
            sample_image = calibrator.get_sample_image()
            calibrator.draw_frame_axes(sample_image, length=6.0)
            cv.imwrite(str(calibrator.path / "axes.jpg"), sample_image)
            main_ref = calibrator.get_main_ref()
            import json
            import numpy as np
            with open(main_ref.obj_path, "r") as f:
                obj_points = json.load(f)
            obj_points = np.array(obj_points, dtype=np.float32)
            print("obj_points", obj_points)
            # obj_points are points around a rectangle in clockwise order
            # fill a grid-like pattern in the rectangle
            grid_num = 7
            grid_pts = []
            for i in range(grid_num):
                for j in range(grid_num):
                    x = obj_points[0][0] + (obj_points[1][0] - obj_points[0][0]) * i / (grid_num - 1)
                    y = obj_points[0][1] + (obj_points[3][1] - obj_points[0][1]) * j / (grid_num - 1)
                    grid_pts.append((x, y))
            grid_pts = np.array(grid_pts, dtype=np.float32)
            grid_pts = grid_pts.reshape(-1, 1, 2)
            # append 0 as z coordinate
            grid_pts = np.concatenate((grid_pts, np.zeros((grid_pts.shape[0], 1, 1), dtype=np.float32)), axis=2)
            print("grid_pts", grid_pts)
            grid_pts = grid_pts.reshape(-1, 3)
            obj_points = np.concatenate((obj_points, grid_pts), axis=0)

            projected_points = cv.projectPoints(
                obj_points, camera.rotation, camera.translation, camera.intrinsic, camera.distortion
            )
            projected_points = projected_points[0].reshape(-1, 2)
            for i in range(len(obj_points)):
                cv.circle(sample_image, tuple(map(int, projected_points[i])), 3, (0, 0, 255), -1)
            cv.imwrite(str(calibrator.path / "projected.jpg"), sample_image)
            camera.summary()
            return
        else:
            if args.base is not None:
                calibrator.path = Path(args.base)
                calibrator.load()
                calibrator.path = Path(args.cam)
            rms = calibrator.calibrate(fix_intrinsic=args.infix, use_intrinsic_guess=args.inguess)
            print(f"[Calibration RMS: {rms}]")
            camera = calibrator.get_camera()
            sample_image = calibrator.get_sample_image()
            calibrator.draw_frame_axes(sample_image)
            cv.imwrite(str(calibrator.path / "axes.jpg"), sample_image)
            camera.summary()
            calibrator.save()
            return
    if args.camset is not None:
        if args.camdir is not None:
            camdir = Path(args.camdir)
        else:
            camdir = Path(args.camset)
        if args.show:
            print("Showing calibration result")
            print(f"Loading camera set from {args.camset}")
            calibrator = CameraSetFileCalibrator(args.camset)
            calibrator.load()
            camset = calibrator.get_camset()
            camset.summary()
            return
        
        camdir.mkdir(exist_ok=True)
        if camdir is None or not camdir.exists() or args.cams is None:
            raise ValueError("camdir and cams should be specified")
        camera_names = args.cams.split(",")
        calibrator = CameraSetFileCalibrator(args.camset)
        for camera_name in camera_names:
            cam_calib = CameraFileCalibrator(camdir / camera_name)
            if not cam_calib.is_pickle_exists():
                raise ValueError(f"{cam_calib.pickle_path} does not exist")
            cam_calib.load().get_camera()
            calibrator.add_camera(cam_calib.get_camera())
        # Perform different fmat computation
        if args.common:
            calibrator.compute_fmat_by_commons()
        else:
            calibrator.compute_fmat_by_cameras()
        camset = calibrator.get_camset()
        camset.summary()
        calibrator.save()
        return


if __name__ == "__main__":
    main()