import numpy as np
import pickle
from pathlib import Path
from .camera import Camera, CameraSet
import json
import cv2 as cv

"""
Camset Folder Structure
Pickles:
    {camset_name}.pickle
    camset.pickle
Notations:
    cameras.txt
    ./refs/{type}.json
    ./refs/{type}_pts.json
    ./commons/{type}.json

    The main {type} is "main", which is the main reference for cameras' rvec, tvec.

The CameraSet can contain the Camera data directly or save to separate files.
the camera should be able to calibrate isolatedly then combined to a CameraSet.
"""

class RefFile:
    obj_path: Path
    def __init__(self, obj_path) -> None:
        self.obj_path = Path(obj_path)
    @property
    def img_path(self):
        return self.obj_path.with_name(self.obj_path.stem + "_pts.json")
    @property
    def type(self):
        return self.obj_path.stem
    def get_obj_data(self):
        with open(self.obj_path, "r") as f:
            return json.load(f)
    def get_pts_data(self):
        with open(self.img_path, "r") as f:
            return json.load(f)
    def is_valid(self):
        if not self.obj_path.exists():
            return False
        if not self.img_path.exists():
            return False
        return True

class CommonFile:
    path: Path
    def __init__(self, path) -> None:
        self.path = Path(path)
    @property
    def type(self):
        return self.path.stem
    
    def is_valid(self):
        try:
            t = self.type
        except Exception as e:
            return False
        return True


class CameraCalibrator:
    def __init__(self, name=None) -> None:
        self.name = name
        self.camera = Camera()
        self.obj_points = []
        self.img_points = []
        self.main_obj_points = None
        self.main_img_points = None
        self.commons = {}
        self.sample_image = None

    def set_camera(self, camera):
        self.camera = camera
    
    def get_camera(self):
        return self.camera

    def add_obj_img_points(self, obj_points, img_points):
        self.obj_points.append(obj_points)
        self.img_points.append(img_points)

    def set_main_obj_img_points(self, obj_points, img_points):
        self.main_obj_points = obj_points
        self.main_img_points = img_points

    def add_common_points(self, category, fid_points):
        self.commons[category] = fid_points

    def set_sample_image(self, image):
        self.sample_image = image

    def calibrate(self, fix_intrinsic=False, use_intrinsic_guess=False):
        if self.main_img_points is None or self.main_obj_points is None:
            raise ValueError("main_img_points or main_obj_points is None")
        obj_points = [np.array([self.main_obj_points], dtype=np.float32)] + self.obj_points
        img_points = [np.array([self.main_img_points], dtype=np.float32)] + self.img_points
        flags = 0
        # flags = cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6 | cv.CALIB_ZERO_TANGENT_DIST
        # flags = cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6 
        criteria = None
        # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000000, 0.01)
        flags |= cv.CALIB_FIX_ASPECT_RATIO
        #flags |= cv.CALIB_THIN_PRISM_MODEL
        #flags |= cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6 | cv.CALIB_ZERO_TANGENT_DIST
        if fix_intrinsic:
            flags |= cv.CALIB_FIX_INTRINSIC
        if use_intrinsic_guess:
            flags |= cv.CALIB_USE_INTRINSIC_GUESS
        if fix_intrinsic or use_intrinsic_guess:
            flags |= cv.CALIB_ZERO_TANGENT_DIST
            #flags |= cv.CALIB_FIX_TANGENT_DIST
            flags |= cv.CALIB_FIX_PRINCIPAL_POINT
            flags |= cv.CALIB_FIX_TAUX_TAUY
            flags |= cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6
            #flags |= cv.CALIB_FIX_S1_S2_S3_S4

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            obj_points,
            img_points,
            self.sample_image.shape[:2][::-1] if self.sample_image is not None else (1920, 1080),
            self.camera.intrinsic,
            self.camera.distortion,
            flags=flags,
            criteria=criteria
        )
        if not ret:
            raise ValueError("Calibration failed")
        
        tvec, rvec = tvecs[0], rvecs[0]
        self.camera.intrinsic = mtx
        self.camera.distortion = dist
        self.camera.extrinsic = np.hstack([cv.Rodrigues(rvec)[0], tvec])
        self.camera.rotation = cv.Rodrigues(rvec)[0]
        self.camera.translation = tvec
        self.camera.projection = mtx @ self.camera.extrinsic
        self.camera.commons = self.commons
        return ret

    def draw_frame_axes(self, image, length=4.5, thickness=2):
        camera = self.camera
        image = cv.drawFrameAxes(image, camera.intrinsic, camera.distortion, camera.rotation, camera.translation, length, thickness)
        return image
        

class CameraFileCalibrator(CameraCalibrator):
    def __init__(self, path) -> None:
        self.path = Path(path)
        if self.path.exists() and not self.path.is_dir():
            raise ValueError(f"{path} is not a directory")
        super().__init__(self.path.stem)

    @property
    def pickle_path(self):
        return self.path / "camera.pickle"

    @property
    def ref_dir(self):
        return self.path / "refs"

    @property
    def common_dir(self):
        return self.path / "commons"
    
    def get_sample_image_path(self):
        potential_suffixes = [".jpg", ".png"]
        potential_filename = ["0", "1"]
        for suffix in potential_suffixes:
            for filename in potential_filename:
                img_path = (self.path / "samples" / filename).with_suffix(suffix)
                if img_path.exists():
                    return img_path
        return None
    
    def get_sample_video_path(self):
        potential_suffixes = [".mov", ".mp4"]
        for suffix in potential_suffixes:
            img_path = (self.path / self.name).with_suffix(suffix)
            if img_path.exists():
                return img_path
            img_path = (self.path.parent / self.name).with_suffix(suffix)
            if img_path.exists():
                return img_path
        return None
    
    def get_sample_image(self):
        if self.sample_image is not None:
            return self.sample_image
        sample_image_path = self.get_sample_image_path()
        if sample_image_path is None:
            return None
        return cv.imread(str(sample_image_path))
    
    def calibrate(self, fix_intrinsic=False, use_intrinsic_guess=False):
        self.camera.name = self.path.stem
        ref_files = self.load_refs()
        main_ref = None
        for rfile in ref_files:
            if rfile.type == "main":
                main_ref = rfile
                break
        if main_ref is None:
            raise Exception("main reference is not found")
        ref_files.remove(main_ref)
        ref_files.insert(0, main_ref)
        print("Ref files:", "\n".join([f"{rfile.obj_path} <-> {rfile.img_path}" for rfile in ref_files]))
        # Load reference files
        # The first one is the main reference
        for i, rfile in enumerate(ref_files):
            rfile: RefFile
            with open(rfile.obj_path, "r") as f:
                obj_points = json.load(f)
            with open(rfile.img_path, "r") as f:
                img_points = json.load(f)
            if i == 0:
                self.set_main_obj_img_points(obj_points, img_points)
            else:
                obj_points, img_points = np.array(obj_points, dtype=np.float32), np.array(img_points, dtype=np.float32)
                print(f"{i} other refs:", obj_points.shape, img_points.shape)
                if obj_points.ndim == 3:
                    for j in range(obj_points.shape[0]):
                        self.add_obj_img_points(obj_points[j], img_points[j])
                else:
                    self.add_obj_img_points(obj_points, img_points)

        # Load common files into Camera, for further use (if CameraSet need it)
        # so it's not involved in calibration here.
        common_files = self.load_commons()
        for cfile in common_files:
            cfile: CommonFile
            with open(cfile.path, "r") as f:
                data = json.load(f)
            self.add_common_points(cfile.type, data)

        self.set_sample_image(cv.imread(str(self.get_sample_image_path())))
        return super().calibrate(fix_intrinsic, use_intrinsic_guess)

    def load_refs(self):
        ref_dir = self.ref_dir
        ref_file_list = []
        for ref_file in ref_dir.glob("*.json"):
            ref_file = RefFile(ref_file)
            if not ref_file.img_path.exists():
                continue
            ref_file_list.append(ref_file)
        return ref_file_list

    def get_main_ref(self):
        for ref_file in self.load_refs():
            if ref_file.type == "main":
                return ref_file
        return None
    
    def load_commons(self):
        common_dir = self.common_dir
        common_file_list = []
        for common_file in common_dir.glob("*.json"):
            common_file = CommonFile(common_file)
            if not common_file.is_valid():
                continue
            common_file_list.append(common_file)
        return common_file_list
    
    def is_pickle_exists(self):
        return self.pickle_path.exists()

    def load(self):
        if self.pickle_path.exists():
            self.camera.load_path(self.pickle_path)
        self.camera.name = self.path.stem
        return self
    
    def save(self):
        print(f"Saving {self.name} to {self.pickle_path}")
        self.camera.save_path(self.pickle_path)


class CameraSetCalibrator:
    def __init__(self) -> None:
        self.camset = CameraSet()

    def add_camera(self, camera):
        self.camset.add_camera(camera)
    
    def compute_fmat_by_cameras(self):
        self.camset.compute_fundamental_matrix_table()

    def compute_fmat_by_commons(self):
        self.camset.compute_fundamental_matrix_table_from_common_points()

    def get_camset(self):
        return self.camset


class CameraSetFileCalibrator(CameraSetCalibrator):
    def __init__(self, path) -> None:
        self.path = Path(path)
        if self.path.exists() and not self.path.is_dir():
            raise ValueError(f"{path} is not a directory")
        self.path.mkdir(exist_ok=True)
        super().__init__()

    @property
    def pickle_path(self):
        return self.path / "camset.pickle"
    
    def get_sample_video_path(self, camera_name):
        potential_suffixes = [".mov", ".mp4"]
        for suffix in potential_suffixes:
            vid_path = (self.path / camera_name).with_suffix(suffix)
            if vid_path.exists():
                return vid_path
        return None
    
    def load(self):
        if self.pickle_path.exists():
            self.camset.load(self.pickle_path)
        return self
    
    def save(self):
        self.camset.save(self.pickle_path)

    def is_pickle_exists(self):
        return self.pickle_path.exists()