import cv2
from cv2 import dnn_superres
import numpy as np

class RoiHandler:
    def __init__(self) -> None:
        self.x_offset = 0
        self.y_offset = 0

    
    def set_xyoffset(self, x_offset, y_offset):
        self.x_offset = x_offset
        self.y_offset = y_offset

    # Define the target ball's roi
    def find_roi(self, frame, x, y, w, h):
        x1, y1, x2, y2 = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        roi = frame[y1:y2, x1:x2]
        return roi


    # Improve image quality during upsampling    
    def enhance_image(self, roi):
        blurred_roi = cv2.GaussianBlur(roi, (3, 3), 0)
        
        resized_roi = cv2.resize(blurred_roi, (100, 100), interpolation=cv2.INTER_LANCZOS4)

        sharpen_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])

        sharpened_roi = cv2.filter2D(resized_roi, -1, sharpen_kernel)
        return sharpened_roi
