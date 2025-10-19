from pathlib import Path
from tqdm import tqdm
import numpy as np
from cv2 import aruco
import cv2 as cv

class ArucoDetectorHandler:
    def __init__(self) -> None:
        aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_4X4_50 )
        arucoParams = aruco.DetectorParameters()
        # arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        arucoParams.cornerRefinementMaxIterations = 300
        arucoParams.cornerRefinementMinAccuracy = 0.01
        arucoParams.cornerRefinementWinSize = 1

        detector = cv.aruco.ArucoDetector(aruco_dict, arucoParams)
        self.aruco_dict = aruco_dict
        self.arucoParams = arucoParams
        self.arucoParams = arucoParams
        self.detector = detector
        self.aruco_unit_offset = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        self.aruco_unit_offset3d = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])

    def detect_markers_in_image(self, img):
        # Return corners, ids, rejectedImgPoints
        detector = self.detector
        img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = detector.detectMarkers(img_gray)
        # when no marker is detected : corners,ids = (),None
        if ids is None:
            ids = []
        return corners, ids, rejectedImgPoints
    
    @staticmethod
    def corners_to_dict(corners, ids):
        id_pts = {}
        for i, marker_id in enumerate(ids):
            marker_id = marker_id[0]
            id_pts[marker_id] = [corners[i][0][k] for k in range(4)]
        return id_pts
        
    def draw_markers(self, img, corners, ids, borderColor=(0, 255, 0)):
        corners = np.array(corners)
        ids = np.array(ids)
        img = cv.aruco.drawDetectedMarkers(image=img, corners=corners, ids=ids, borderColor=borderColor)
        return img