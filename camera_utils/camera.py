import numpy as np
import pickle
import itertools
import cv2 as cv

"""
    Camera Serial Format
    intrinsic, (= mtx in opencv)
    distortion,
    extrinsic,
    rotation
    translation

    projection = intrinsic * extrinsic

    The most important design is the camera class should be able to be 
    load-edit-save or load-use
    the CameraSet class should achieve the same goal
"""

class Camera:
    def __init__(self, name=None) -> None:
        self.name = name
        self.ndarray_attr_names = [
            'intrinsic',
            'distortion',
            'extrinsic',
            'rotation',
            'translation',
            'projection',
        ]
        self.attr_names = [
            'commons'
        ]
        self.intrinsic = None
        self.distortion = None
        self.extrinsic = None
        self.rotation = None
        self.translation = None
        self.projection = None
        self.commons = None

    def load_path(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.load_attr_dict(data)

    def get_attr_dict(self):
        data = {}
        for attr_name in self.ndarray_attr_names:
            if not hasattr(self, attr_name):  # Numpy objects
                continue
            v = getattr(self, attr_name)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            data[attr_name] = v
        for attr_name in self.attr_names:  # Python objects
            if not hasattr(self, attr_name):
                continue
            v = getattr(self, attr_name)
            data[attr_name] = v
        return data
    
    def load_attr_dict(self, data):
        for attr_name in self.ndarray_attr_names:
            v = data.get(attr_name)
            if v is not None:
                v = np.array(v)
            setattr(self, attr_name, v)
        for attr_name in self.attr_names:
            v = data.get(attr_name)
            setattr(self, attr_name, v)
        if self.commons is not None:
            for common_type, fid_points in self.commons.items():
                self.commons[common_type] = {int(k): v for k, v in fid_points.items()}
    
    def save_path(self, path):
        data = self.get_attr_dict()
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def get_projection(self):
        if self.projection is None:
            if self.intrinsic is None or self.extrinsic is None:
                raise ValueError("projection is None and (intrinsic or extrinsic is None)")
            self.projection = self.intrinsic @ self.extrinsic
        return self.projection
    
    def summary(self):
        print(f"--- Camera {self.name} ---")
        print(f"Intrinsic matrix \n{self.intrinsic}")
        print(f"Distortion matrix \n{self.distortion}")
        print(f"Extrinsic matrix \n{self.extrinsic}")
        print(f"Projection matrix \n{self.projection}")
        common_cnts = {k: len(v) for k, v in self.commons.items()}
        print(f"Commons \n{common_cnts}")

class CameraSet:
    def __init__(self) -> None:
        self.cameras = []
        self.name_camera_map = {}
        self.fmat_table = None
        self.ndarray_attr_names = []

    def add_camera(self, camera):
        self.cameras.append(camera)
        self.name_camera_map[camera.name] = camera

    def get_camera(self, name):
        return self.name_camera_map.get(name)
    
    def get_fmat(self, c1, c2):
        if self.fmat_table is None:
            # raise ValueError("fmat_table is None")
            return None
        return self.fmat_table.get((c1.name, c2.name))
    
    def get_epiline(self, c1, c2, pt, F, whichImage=1):
        if F is None:
            return None
        line = cv.computeCorrespondEpilines(pt.reshape(-1, 1, 2), whichImage, F).reshape(-1, 3)[0]
        return line
    
    def get_epilines(self, c1, c2, pt, F, whichImage=1):
        if F is None:
            return None
        lines = cv.computeCorrespondEpilines(pt.reshape(-1, 1, 2), whichImage, F).reshape(-1, 3)
        return lines

    def compute_fundamental_matrix_table(self):
        self.fmat_table = {}
        for c1, c2 in itertools.permutations(self.cameras, 2):
            if c1.name is None or c2.name is None:
                continue
            self.fmat_table[(c1.name, c2.name)] = self.compute_fundamental_from_cameras(c1, c2)
        return self.fmat_table
    
    def compute_fundamental_matrix_table_from_common_points(self):
        # This function updates the fmat_table because some common points may not be found
        self.fmat_table = self.fmat_table or {}
        for c1, c2 in itertools.permutations(self.cameras, 2):
            if c1.name is None or c2.name is None:
                continue
            F = self.compute_fundamental_from_common_points(c1, c2)
            if F is None:
                continue
            self.fmat_table[(c1.name, c2.name)] = F

    def compute_fundamental_from_cameras(self, camera1, camera2):
        p1, p2 = camera1.get_projection(), camera2.get_projection()
        F = self.fundamental_from_projections(p1, p2)
        return F

    def compute_fundamental_from_common_points(self, camera1, camera2):
        pts1, pts2 = self.get_common_points(camera1, camera2)
        if len(pts1) < 8:
            # raise ValueError("Not enough common points")
            return None
        F = self.fundamental_from_common_points(pts1, pts2)
        return F

    def fundamental_from_projections(self, p1, p2):
        X, Y = [None] * 3, [None] * 3
        X[0] = np.vstack((p1[1], p1[2]))
        X[1] = np.vstack((p1[2], p1[0]))
        X[2] = np.vstack((p1[0], p1[1]))
        Y[0] = np.vstack((p2[1], p2[2]))
        Y[1] = np.vstack((p2[2], p2[0]))
        Y[2] = np.vstack((p2[0], p2[1]))
        XY = None
        F = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                XY = np.vstack((X[j], Y[i]))
                F[i, j] = np.linalg.det(XY)
        return F
    
    def fundamental_from_common_points(self, pts1, pts2):
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT)
        return F
    
    def get_common_points(self, camera1, camera2):
        pts1, pts2 = [], []
        for category, fid_points1 in camera1.commons.items():
            if category not in camera2.commons:
                continue
            fid_points2 = camera2.commons[category]
            for fid, pt1 in fid_points1.items():
                if fid not in fid_points2:
                    continue
                pt2 = fid_points2[fid]
                pts1.append(pt1)
                pts2.append(pt2)
        return np.array(pts1), np.array(pts2)
    
    def get_attr_dict(self):
        data = {}
        data['cameras'] = {cam.name: cam.get_attr_dict() for cam in self.cameras}
        for attr_name in self.ndarray_attr_names:
            if not hasattr(self, attr_name):
                continue
            v = getattr(self, attr_name)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            data[attr_name] = v
        FMAT_KEY = "fmat_table"
        if self.fmat_table is not None:
            data[FMAT_KEY] = {}
            for (c1, c2), F in self.fmat_table.items():
                data[FMAT_KEY][(c1, c2)] = F.tolist()
        return data
    
    def load_attr_dict(self, data):
        for cname, cam_data in data['cameras'].items():
            cam = Camera(cname)
            cam.load_attr_dict(cam_data)
            self.add_camera(cam)
        for attr_name in self.ndarray_attr_names:
            v = data.get(attr_name)
            if v is not None:
                v = np.array(v)
            setattr(self, attr_name, v)
        FMAT_KEY = "fmat_table"
        if FMAT_KEY in data:
            self.fmat_table = {}
            for (c1, c2), F in data[FMAT_KEY].items():
                self.fmat_table[(c1, c2)] = np.array(F)
    
    def save(self, path):
        data = self.get_attr_dict()
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.load_attr_dict(data)

    def summary(self):
        print(f"--- CameraSet ---")
        for cam in self.cameras:
            cam.summary()
        if self.fmat_table is not None:
            print(f"--- Fundamental Matrix Table ---")
            for (c1, c2), F in self.fmat_table.items():
                print(f"{c1} -> {c2}")
                print(F)
                print("")