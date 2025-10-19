import cv2 as cv
import numpy as np
import torch


def letterbox(im, imgsz):
    # imgsz is list [w, h]
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    r = min(imgsz[0] / shape[1], imgsz[1] / shape[0])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (imgsz[0] - new_unpad[0]) / 2  # width padding
    dh = (imgsz[1] - new_unpad[1]) / 2  # height padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=[0, 0, 0])
    return im

def inverse_letterbox_xywh(xywh, imgsz, orig_size):
    # xywh is list [x, y, w, h] in letterbox coordinate
    # letterbox_size is list [w, h]
    # orig_size is list [w, h]
    shape = [orig_size[1], orig_size[0]]  # [height, width]
    r = min(imgsz[0] / shape[1], imgsz[1] / shape[0])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (imgsz[0] - new_unpad[0]) / 2  # width padding
    dh = (imgsz[1] - new_unpad[1]) / 2  # height padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    x, y, w, h = xywh
    x = (x - left) / new_unpad[1] * shape[1]
    y = (y - top) / new_unpad[0] * shape[0]
    w = w / new_unpad[1] * shape[1]
    h = h / new_unpad[0] * shape[0]
    return [x, y, w, h]

def ndarray_to_tensor(im, device):
    # im = np.stack(self.pre_transform(im))
    if im.ndim == 3:  # add batch dimension
        # im = np.expand_dims(im, axis=0)
        im = np.stack([im])
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
    # im = im[..., ::-1].transpose((2, 0, 1))
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im)
    im = im.to(device)
    im = im.float() / 255
    # print("im shape", im.shape)
    return im