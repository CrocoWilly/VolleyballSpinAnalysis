"""
    No more complex design patterns, just simple classes to hold data,
    the pipeline pass the FrameResult object to next module
    Plan:
    Thread 1: Get frames and track balls
    Thread 2: Run analysis on the results
    Thread 3: Draw results to video

    the term "syncer" is used to check if camera works of a frame are all done,
    the latest camera will receive True from syncer and it should emit the on_result signal.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
# torch.backends.cudnn.enabled = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.allow_tf32 = False

import cv2 as cv
from ultralytics import YOLO
import argparse
from pathlib import Path
import time
from typing import List
from threading import Thread
from blinker import Signal
from queue import Queue
import numpy as np
from itertools import product
from camera_utils.camera import Camera, CameraSet
from camera_utils.calibration import CameraSetFileCalibrator
from scipy.optimize import linear_sum_assignment
from loguru import logger
import re
import visuals
import geometry
import matplotlib.pyplot
import measure
import enum
import math
import datetime
import argparse
import sys
import gc
import imutils
from functools import partial
from pipeline import PipelineStage, ProcessPipelineStage, NullQueue
from vidgear.gears import CamGear, VideoGear, WriteGear
import multiprocessing
from arrayqueues.shared_arrays import ArrayQueue
import json
import threading

### ----------------------------------------------- New Add ------------------------------------------------------- ###
import cv2
import roiPreprocesser
### --------------------------------------------------------------------------------------------------------------- ###


LARGE_NUM = 999999
VIDEOFILE_PATTERN = r"HDR80_(?P<camera>[^_]*)_Live_(?P<year>.{4})(?P<date>.{4})_(?P<time>.{6})_000"
VIDEOFILE_REGEX = re.compile(VIDEOFILE_PATTERN)

def hdr80_match_to_datetime(match):
    year = match.group("year")
    date = match.group("date")
    time = match.group("time")
    date = datetime.datetime.strptime(f"{year} {date} {time}", r"%Y %m%d %H%M%S")
    return date

def get_device_capture(index, fps, width, height):
    fps = int(fps)
    width = int(width)
    height = int(height)
    options = {
        "CAP_PROP_FRAME_WIDTH": width,
        "CAP_PROP_FRAME_HEIGHT": height,
        "CAP_PROP_FPS": fps,
        'THREADED_QUEUE_MODE': False,
    }
    print(f"VidGeear cap options \n{options}")
    cap = VideoGear(source=index, logging=False, **options).start()
    return cap

def __get_rally_video_name_from_datetime(dt, camera_name):
    date_str = dt.strftime(r"%Y%m%d")
    time_str = dt.strftime(r"%H%M%S")
    return f"Rally_{camera_name}_{date_str}_{time_str}"

def get_rally_video_name_from_datetime(start_dt, elapsed_seconds, camera_name):
    date_str = start_dt.strftime(r"%Y%m%d")
    time_str = start_dt.strftime(r"%H%M%S")
    return f"Rally_{date_str}_{time_str}-{int(elapsed_seconds):03d}_{camera_name}"

def get_rally_dir_from_datetime(start_dt, elapsed_seconds):
    date_str = start_dt.strftime(r"%Y%m%d")
    time_str = start_dt.strftime(r"%H%M%S")
    return f"Rally_{date_str}_{time_str}-{int(elapsed_seconds):03d}"


def format_array(arr):
    if arr is None:
        return None
    return f"[{', '.join([f'{a:.2f}' for a in arr])}]"

def try_tolist(obj):
    try:
        return obj.tolist()
    except:
        return obj

class SourceType(enum.Enum):
    FILE = 'file'
    DEVICE = 'device'

class SourceInfo:
    """
        This class should guarantee that it would work after pickled for multiprocessing.
    """
    def __init__(self, source, fps, width, height, source_type=SourceType.FILE):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.source_type = source_type

def build_source(source_info: SourceInfo):
    print(f"build_source: {source_info.source_type} {source_info.source}")
    print(f"info: {source_info.fps} {source_info.width} {source_info.height}")
    if source_info.source_type == SourceType.FILE:
        options = {}
        cap = VideoGear(source=source_info.source, logging=False, **options).start()
        return cap
    elif source_info.source_type == SourceType.DEVICE:
        # options = {
        #     "CAP_PROP_FRAME_WIDTH": source_info.width,
        #     "CAP_PROP_FRAME_HEIGHT": source_info.height,
        #     "CAP_PROP_FPS": source_info.fps,
        #     'THREADED_QUEUE_MODE': True,
        # }
        # cap = VideoGear(source=source_info.source, logging=False, **options).start()
        print(f"get_device_capture: {source_info.source} {source_info.fps} {source_info.width} {source_info.height}")
        source = int(source_info.source)
        # time.sleep(source + 3)
        cap = get_device_capture(source, source_info.fps, source_info.width, source_info.height)
        return cap
    else:
        raise ValueError(f"Unknown source type {source_info.source_type}")
        

class Track2D:  # wrap the ultralytics result from id per frame to list of bbox
    def __init__(self, camera, track_id):
        self.track_id = track_id
        self.camera = camera
        self.frame_id_xywh = {}
        self.last_frame_id = None

    def add_frame(self, frame_id, xywh):
        self.frame_id_xywh[frame_id] = np.array(xywh)
        self.last_frame_id = frame_id if self.last_frame_id is None else max(self.last_frame_id, frame_id)

    def xywh(self, frame_id=None):
        if frame_id is None:
            frame_id = self.last_frame_id
        return self.frame_id_xywh.get(frame_id)

    def xy(self, frame_id=None):
        if frame_id is None:
            frame_id = self.last_frame_id
        xywh = self.xywh(frame_id)
        if xywh is None:
            return None
        return xywh[:2] - (xywh[2:] / 2)
    
    def recent_xywhs(self, frame_id=None, n=10):
        if frame_id is None:
            frame_id = self.last_frame_id
        xywh_list = []
        for i in range(n):
            xywh = self.xywh(frame_id - i)
            if xywh is not None:
                xywh_list.append(xywh)
        return xywh_list

class Track3D:
    def __init__(self, track_id) -> None:
        self.track_id = track_id
        self.frame_id_pos3ds = {}
        self.frame_id_cam_track2ds = {}
        self.last_frame_id = None

    def add_frame(self, frame_id, pos3d, cam_track2ds):
        self.frame_id_pos3ds[frame_id] = pos3d
        self.frame_id_cam_track2ds[frame_id] = cam_track2ds # cam:track2d
        self.last_frame_id = frame_id if self.last_frame_id is None else max(self.last_frame_id, frame_id)

    def pos3d(self, frame_id=None):
        if frame_id is None:
            frame_id = self.last_frame_id
        return self.frame_id_pos3ds.get(frame_id)
    
    def cam_track2ds(self, frame_id=None):
        if frame_id is None:
            frame_id = self.last_frame_id
        return self.frame_id_cam_track2ds.get(frame_id)
    
    def recent_pos3ds(self, frame_id=None, n=10):
        if frame_id is None:
            frame_id = self.last_frame_id
        pos3d_list = []
        for i in range(n):
            pos3d = self.pos3d(frame_id - i)
            if pos3d is not None:
                pos3d_list.append(pos3d)
        return pos3d_list


# Hold results from each camera
class FrameResult:
    def __init__(self, frame_id):
        self.frame_id = frame_id  # This is absolute frame id since system started
        self.cameras = None
        self.cameraset = None
        self.camera_frames = dict()
        self.camera_result_frames = dict()
        self.rally_dir = None
        self.camera_output_paths = dict()
        self.camera_results = dict()
        self.camera_track2ds = dict()
        self.camera_detections = dict()
        self.camera_active_track2ds = dict() # updated every frame, the valid and alive track2ds 
        self.camera_filtered_track2ds = dict()  #  the invalid alive track2ds (maybe because it's static)
        self.camera_epilines = dict()
        self.track3ds_list = []
        self.rally_tracker_state = None
        self.main_pos3d = None
        self.main_color = None
        self.main_track3d = None
        self.ball_vel = None
        self.ball_acc = None
        self.ball_jer = None

        ### ----------------------------------------------- New Add ------------------------------------------------------- ###
        self.spin_rate_rpm = None
        ### --------------------------------------------------------------------------------------------------------------- ###

        self.ball_vel_kmh = None
        self.ball_acc_ms2 = None
        self.ball_jer_ms3 = None
        self.ball_tilt_angle_deg = None
        self.ball_yaxis_angle_deg = None
        self.court_3d_figure = None
        self.court_plane_figure = None
        self.court_3d_image = None
        self.court_plane_image = None

        self.event_is_rally_begin = None
        self.event_is_rally = None
        self.event_is_rally_end = None
        self.event_is_rally_reset = None
        self.event_is_collide = None
        self.event_is_serve = None
        self.event_is_attack = None
        self.event_is_spike = None
        self.event_is_serve_cross_net = None
        self.event_is_serve_receive = None
        self.serve_speed_kmh = None
        self.serve_cross_net_seconds = None
        self.spike_speed_kmh = None
        self.spike_height_m = None
        self.attack_speed_kmh = None
        self.attack_height_m = None
        self.serve_start_pos3d = None
        self.serve_end_pos3d = None

        self.start_time = None
        self.fps = None

class RateLimiter:
    def __init__(self, interval):
        self.interval = interval
        self.last_time = None

    def wait(self):
        if self.last_time is None:
            self.last_time = time.time()
            return
        elapsed = time.time() - self.last_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_time = time.time()

class CameraReader:
    """
        This class is used to read frames from multiple cameras, and aggregate the frames into a FrameResult object.
    """

    on_result = Signal(FrameResult)
    def __init__(self, cameras, camera_source_infos, start_time, fps=60, simulate_fps=None, is_process=False) -> None:
        self.start_time = start_time
        # Set simulated_fps to None to disable simulation 
        self.simulate_fps = simulate_fps
        self.simulate_interval = None
        if simulate_fps is not None:
            self.simulate_interval = 1 / simulate_fps
        
        self.fps = fps
        self.cameras = cameras
        self.camera_source_infos = camera_source_infos
        self.source_threads = {}
        self.source_queues = {}
        self.is_process = is_process
        logger.info(f"CameraReader: is_process={is_process}")
        if is_process:
            self.sync_barrier = multiprocessing.Barrier(len(camera_source_infos))
        else:
            self.sync_barrier = threading.Barrier(len(camera_source_infos))
        self._sync_diff_dict = {_ : list() for _ in range(len(camera_source_infos))}
        for source_id, source_info in enumerate(camera_source_infos.values()):
            if source_info is None:
                raise ValueError(f"Camera source is None")
            if is_process:
                # Multi-processing
                q = ArrayQueue(4000)
                t = multiprocessing.Process(target=self.source_thread_fucnt, args=(source_id, source_info, q, self.sync_barrier, self._sync_diff_dict, -5))
            else:
                # Multi-threading
                q = Queue(maxsize=60 * 20)
                t = Thread(target=self.source_thread_fucnt, args=(source_id, source_info, q, self.sync_barrier, self._sync_diff_dict))
            self.source_threads[source_info] = t
            self.source_queues[source_info] = q
        self.last_frame_id = 0
        for t in self.source_threads.values():
            t.start()

    @staticmethod
    def source_thread_fucnt(source_id, source_info, queue, sync_barrier, _sync_diff_dict, nice=None):
        if nice is not None:
            try:
                os.nice(nice)
                logger.info(f"CameraReader: Set nice value to {nice}")
            except Exception as e:
                logger.error(f"CameraReader: Failed to set nice value: {e}")
        source = build_source(source_info)
        fid = 0
        while True:
            fid += 1
            # Barrier for sync
            try:
                bid_time = time.time()
                bid = sync_barrier.wait()
                # logger.debug(f"Barrier wait Fid={fid} (BarrierId={bid}) waited in {time.time() - bid_time} (FPS {1/(time.time() - bid_time)})")
                # bid = source_id
            except threading.BrokenBarrierError:
                # Other thread will use broke barrier to stop the thread
                queue.put(np.array([0], dtype=float))
                logger.debug(f"Cap read Fid={fid} (BarrierId={bid}) is broken")
                break

            st_time = time.time()
            frame = source.read()
            _sync_diff_dict[bid].append(st_time)
            if frame is None:
                if not sync_barrier.broken:
                    sync_barrier.abort()
                    logger.debug(f"Cap read Fid={fid} (BarrierId={bid}) is None, aborting")
                else:
                    logger.debug(f"Cap read Fid={fid} (BarrierId={bid}) is None, already broken")
                queue.put(np.array([0], dtype=float))
                logger.debug(f"Cap read Fid={fid} (BarrierId={bid}) is None, broken")
                break
            # logger.debug(f"Cap read Fid={fid} (BarrierId={bid} at {st_time}), qlen {queue.qsize()} in {time.time() - st_time} (FPS {1/(time.time() - st_time)})")
            while True:  # spin lock
                # Because the arrayqueue will throw exception if full, so we need to retry until it can put.
                # This usually happens in FILE source, because the file source is too fast (comparing to DEVICE).
                try:
                    queue.put(frame)
                    break
                except Exception as e:
                    pass
        # fid -= 1
        # if source_id == 0:
        #     # sync diff analysis
        #     acc_diff = 0
        #     max_diff = 0
        #     for i in range(fid):
        #         t1 = _sync_diff_dict[0][i]
        #         t2 = _sync_diff_dict[1][i]
        #         acc_diff += abs(t1 - t2)
        #         max_diff = max(max_diff, abs(t1 - t2))
        #     logger.info(f"CameraReader: Sync Diff Total: {acc_diff}, Max: {max_diff}, Average: {acc_diff/fid}s (numfs={fid})")
        source.stop()

    def run(self):
        is_ended = False
        while True:
            st_time = time.time()
            self.last_frame_id += 1
            frame_result = FrameResult(self.last_frame_id)
            frame_result.start_time = self.start_time
            frame_result.fps = self.fps
            frame_result.cameras = self.cameras
            frame_result.camera_frames = {}
            for cam, source_info in self.camera_source_infos.items():
                sim_st_time = time.time()
                source_st_time = time.time()
                frame = self.source_queues[source_info].get()                
                # logger.info(f"Frame {self.last_frame_id} source read in {time.time() - source_st_time:.2f}s, FPS: {1/(time.time() - source_st_time):.2f}")
                if frame is None or len(frame) == 1: # the arrayqueue can't store None, so use array([1]) as None instead.
                    self.on_result.send(None)
                    logger.info(f"Frame {self.last_frame_id} CameraReader is ended (signal sended)")
                    is_ended = True
                    break
                frame_result.camera_frames[cam] = frame
                frame_result.camera_result_frames[cam] = frame.copy()
            if self.simulate_interval is not None:
                wait_time = max(0, self.simulate_interval - (time.time() - sim_st_time))
                if wait_time > 0:
                    time.sleep(wait_time)
            if is_ended:
                break
            # logger.info(f"Frame {self.last_frame_id} read in {time.time() - st_time:.2f}s, FPS: {1/(time.time() - st_time):.2f}")
            self.on_result.send(frame_result)

class ThreadDetector:
    def __init__(self, model_path, device, imgsz, input_queue, output_queue, done_syner, threads_per_device=8, det_log_level=None):
        # self.model = model
        self.model_path = model_path
        self.model = None
        self.device = device
        self.imgsz = imgsz
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.det_log_level = det_log_level
        is_use_process = True
        self.is_use_process = is_use_process
        self.done_syner = done_syner
        # self.tensor_queue = Queue(maxsize=30)
        if is_use_process:
            self.process_input_queue = torch.multiprocessing.Queue(maxsize=3)
            self.process_output_queue = torch.multiprocessing.Queue(maxsize=3)
            self.process_frame_input_queue = ArrayQueue(300)
            
            self.run_process = torch.multiprocessing.Process(target=self.run_funct, args=(model_path, device, imgsz, self.process_input_queue, self.process_frame_input_queue, self.process_output_queue,
                                                                                          self.det_log_level))
            self.move_in_thread = Thread(target=self.queue_send_funct, args=(self.input_queue, self.process_frame_input_queue, self.process_input_queue))
            self.move_out_thread = Thread(target=self.queue_receive_funct, args=(self.process_output_queue, self.output_queue, self.done_syner))
            self.move_in_thread.start()
            self.move_out_thread.start()
            self.run_process.start()
        else:
            self.run_thread = Thread(target=self.run_funct, args=(self.input_queue, self.output_queue))
            self.run_thread.start()
        self.infer_count = 0

    def queue_send_funct(self, input_queue, output_frame_queue, output_queue):
        while True:
            data = input_queue.get()
            if data is None:
                output_queue.put(None)
                break
            data_id, frame = data
            logger.debug(f"ThreadDetector: data_id={data_id} frame sended to process")
            output_queue.put(data_id)
            output_frame_queue.put(frame)
        logger.debug(f"ThreadDetector: queue_send_funct stopped")

    def queue_receive_funct(self, input_queue, output_queue, done_syncer):
        while True:
            data = input_queue.get()
            if data is None:
                if done_syncer.get():
                    output_queue.put(None)
                break
            output_queue.put(data)
        logger.debug(f"ThreadDetector: queue_receive_funct stopped")

    @staticmethod
    def run_funct(model_path, device, imgsz, input_queue, input_frame_queue, output_queue, log_level=None):
        model = YOLO(model_path)
        model.to(device)
        infer_count = 0
        if log_level is not None:
            logger.remove()  # remove the old handler. Else, the old one will work along with the new one you've added below'
            logger.add(sys.stdout, level=log_level)
        while True:
            # data = input_queue.get()
            data_id = input_queue.get()
            if data_id is None:
                output_queue.put(None)
                break
            frame = input_frame_queue.get()
            # if data is None:
            # data_id, frame = data  # data_id can be anything that can be key of dict
            st_time = time.time()
            # result = model.predict(frame, verbose=False, imgsz=imgsz, half=False, device=device, conf=0.03)[0]
            # result = model.predict(frame, verbose=False, imgsz=imgsz, half=False, device=device, conf=0.02)[0]
            result = model.predict(frame, verbose=False, imgsz=imgsz, half=False, device=device, conf=0.0075)[0]
            infer_count += 1
            # if (1/(time.time() - st_time)) > 30:
            #     logger.debug(f"ThreadDetector: Device: {device} detected in {time.time() - st_time:.2f}s FPS: {1/(time.time() - st_time):.2f}")
            # else:
            #     logger.warning(f"ThreadDetector: Device: {device} detected in {time.time() - st_time:.2f}s FPS: {1/(time.time() - st_time):.2f} detectlowfps")
            detections = []
            boxes = result.boxes
            for xywh, class_id, conf in zip(boxes.xywh, boxes.cls, boxes.conf):
                detections.append((xywh.tolist(), class_id.item(), conf.item()))
            result = detections
            output_queue.put((data_id, result))
        logger.info(f"ThreadDetector: Device: {device} stopped, infer_count={infer_count}")

    def join(self):
        if self.is_use_process:
            self.move_in_thread.join()
            self.move_out_thread.join()
            self.run_process.join()
        else:
            self.run_thread.join()

class ThreadOrderedQueue:
    def __init__(self, input_queue) -> None:
        self.input_queue = input_queue
        self.result_dict = {}

    def get(self, wanted_data_id):
        logger.debug(f"ThreadOrderedQueue: wanted_data_id={wanted_data_id} requiring, resultlen={len(self.result_dict)}")
        while wanted_data_id not in self.result_dict:
            data = self.input_queue.get()
            data_id, result = data
            logger.debug(f"ThreadOrderedQueue: data_id={data_id} result received, resultlen={len(self.result_dict)}")
            self.result_dict[data_id] = result
        return self.result_dict.pop(wanted_data_id)
        
class BallDetector:
    on_result = Signal(FrameResult)
    # input_queue -> output_queue -> ordered_queue
    def __init__(self, model_path, model_imgsz, devices=None, thread_per_device=1, det_log_level=None):
        self.imgsz = model_imgsz
        self.queue_size = 5
        self.thread_per_device = thread_per_device
        if devices is None:
            devices = [i for i in range(torch.cuda.device_count())]
        self.thread_detectors = []
        self.input_queue = Queue(maxsize=self.queue_size)
        # because key need to be consistent even if pickled (the queue will pickle it behind the scene), 
        # so use int as key, and a map to get the data of key.
        self.next_key = 0
        self.key_data_map = {}  
        self.key_queue = Queue(maxsize=self.queue_size)
        self.result_queue = Queue(maxsize=self.queue_size)
        self.ordered_queue = ThreadOrderedQueue(self.result_queue)
        self.frame_result_syncer = {}  # a queue for each frame_result for checking if the frame_result is done
        self.done_syncer = torch.multiprocessing.Queue()
        logger.debug(f"Pytorch read cuda version: {torch.version.cuda}")
        for device in devices:
            device_name = torch.cuda.get_device_name(device)
            for i in range(thread_per_device):
                # model = YOLO(model_path)
                device = torch.device(f"cuda:{device}")
                # model.to(device)
                detector = ThreadDetector(model_path, device, model_imgsz, self.input_queue, self.result_queue, self.done_syncer,
                                          det_log_level=det_log_level)
                self.thread_detectors.append(detector)
                logger.debug(f"BallDetector: Device {device} Thread {i} started ({device_name})")
        self.run_detector_thread = Thread(target=self.run_detector, args=(self.key_queue,))
        self.run_detector_thread.start()

    def on_frame(self, frame_result):
        if frame_result is None:
            self.key_queue.put(None)
            for i in range(len(self.thread_detectors)):
                self.input_queue.put(None)
                if i != (len(self.thread_detectors) - 1):
                    self.done_syncer.put(False)
                else:
                    self.done_syncer.put(True)
            logger.info(f"BallDetector: Frame is None, joining all threads")
            for td in self.thread_detectors:
                td.join()
            logger.info(f"BallDetector: all threads stopped")
            return
        logger.info(f"BallDetector Frame {frame_result.frame_id} received, bdqlen={self.input_queue.qsize()}/{self.input_queue.maxsize}")
        self.frame_result_syncer[frame_result] = fq = Queue()
        for i in range(len(frame_result.camera_frames) - 1):
            fq.put(False)
        fq.put(True)
        for cam, frame in frame_result.camera_frames.items():
            # key = (frame_result, cam)
            key = self.next_key
            self.next_key += 1
            self.key_data_map[key] = (frame_result, cam)
            self.input_queue.put((key, frame))  # for model threads
            self.key_queue.put(key)  # for consumer to get keys, which is (FrameResult, Camera) tuple.
    
    def run_detector(self, key_queue):
        while True:
            key = key_queue.get()
            if key is None:
                # if self.done_syncer.get():
                if True:
                    logger.info(f"BallDetector all done, send None to on_result")
                    self.on_result.send(None)
                break
            frame_result, cam = self.key_data_map[key]
            del self.key_data_map[key]  # !!important!! to prevent memory leak
            # result_tuple = self.ordered_queue.get(key)
            # result, detections = result_tuple
            detections = self.ordered_queue.get(key)
            logger.debug(f"BallDetector: Frame {frame_result.frame_id} Camera {cam.name} detections received \n{detections}")
            st_time = time.time()
            frame_result: FrameResult
            # frame_result.camera_results[cam] = result
            frame_result.camera_detections[cam] = detections.copy()
            if frame_result.frame_id % 100 == 0:
                # logger.info(f"Frame {frame_result.frame_id}: Camera {cam.name} detected in {time.time() - st_time:.2f}s FPS: {1/(time.time() - st_time):.2f}")
                pass

            if self.frame_result_syncer[frame_result].get():
                del self.frame_result_syncer[frame_result]  # !!important!! to prevent memory leak
                self.on_result.send(frame_result)
                logger.info(f"Frame {frame_result.frame_id} all cameras detected send")

# This object's life cycle is eternal
class BallTracker:
    """
        Functionality: Compute the physical properties of the ball, such as position, velocity, acceleration, jerk, speed, angle, etc.
        And determine the main ball.
    """
    on_result = Signal(FrameResult)
    def __init__(self, cameraset, cameras, fps=60, imgsz=(960, 960), input_size=(1920, 1080)):
        self.cameraset: CameraSet = cameraset
        self.cameras = cameras
        self.fps = fps
        self.input_size = input_size  # the image size given to ultralytics might be smaller, the bbox should corrected here.
        self.imgsz = imgsz # the model input size
        self.cam_track2ds = {cam: dict() for cam in cameras}
        self.track3ds_list = []
        self.next_track3d_id = 0
        self.ball_pos_var = measure.ObserveVariable(smooth=0.5, skip=1)
        self.ball_vel_var = measure.ObserveVariable(smooth=0.5)
        self.ball_acc_var = measure.ObserveVariable()
        self.track_queue = Queue(maxsize=10)
        self.first_stage = PipelineStage(self.process_frame_result, input_queue=self.track_queue)
        self.middle_stage = PipelineStage(self.process_crossview_lsa_3d_tracking, input_queue=self.first_stage.output_queue)
        self.track_thread = Thread(target=self.track_thread_funct, args=(self.middle_stage.output_queue,))
        self.track_thread.start()
        self.camera_next_track2d_ids = {cam: 0 for cam in cameras}
        ### ----------------------------------------------- New Add ------------------------------------------------------- ###
        ## Optical Flow Parameters ##
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)                                    # ShiTomasi corner detection parameters
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))   # Parameters for Lucas-Kanade optical flow
    
        self.old_gray = []
        self.prev_points = []

        self.frame_interval = 3                     # every 3 frame, calculate the spin rate and find a new set of optical flow points
        self.old_spin_rate = 0
        self.temp_spin_rate = 0
        ### --------------------------------------------------------------------------------------------------------------- ###
    
    ### ----------------------------------------------- New Add ------------------------------------------------------- ###
    # Calculate the spin by Phy-OptiCoord method designed by Yen-Chang Chen 
    def calculate_spin_rate(self, prev_points, curr_points, frame_time):
        
        ## STEP1. Regard the 100*100 bounding box as a coordinate system ##
        # (i.e., ball center = O(0, 0), and each detected optical point = (x, y) in the coordinate system)
        ball_center = [50, 50]
        # Compute the coordinate of each point relative to the ball's center -> regard ball center as (0, 0)
        coordinates_prev = prev_points - ball_center
        coordinates_curr = curr_points - ball_center


        ## STEP2. Use arctan(y/x) to find the theta of each point on the coordinate system ##
        angles_prev = np.arctan2(coordinates_prev[:, 1], coordinates_prev[:, 0])
        angles_curr = np.arctan2(coordinates_curr[:, 1], coordinates_curr[:, 0])


        ## STEP3. Calculate the rotation amount of each point from frame #n-3 to frame #n by using "the predefined quadrant rotational rules" ##
        rotations_quantity = []
        for i in range(len(coordinates_prev)):
            coord_prev = coordinates_prev[i]
            coord_curr = coordinates_prev[i]

            rotation_quantity = 0

            ##    The predefined quadrant rotational rules    ##
            # ---------  同象限內分析: 0rpm ~ 300rpm --------- #
            # case 1: 同象限內轉        theta = abs(cur - prev)
            if( (coord_prev[0] > 0 and coord_prev[1] > 0 and coord_curr[0] > 0 and coord_curr[1] > 0) or 
                (coord_prev[0] < 0 and coord_prev[1] > 0 and coord_curr[0] < 0 and coord_curr[1] > 0) or
                (coord_prev[0] < 0 and coord_prev[1] < 0 and coord_curr[0] < 0 and coord_curr[1] < 0) or
                (coord_prev[0] > 0 and coord_prev[1] < 0 and coord_curr[0] > 0 and coord_curr[1] < 0) ):
                rotation_quantity = abs(angles_curr[i] - angles_prev[i])
            
            # ---------  跨一象限分析: 0rpm ~ 600rpm --------- #
            # case 2: 一四(四一)象限轉  theta = abs(cur - prev)
            elif( (coord_prev[0] > 0 and coord_prev[1] > 0 and coord_curr[0] > 0 and coord_curr[1] < 0) or 
                  (coord_prev[0] > 0 and coord_prev[1] < 0 and coord_curr[0] > 0 and coord_curr[1] > 0) ):
                rotation_quantity = abs(angles_curr[i] - angles_prev[i])

            # case 3: 三四(四三)象限轉  theta = 180度 - abs(cur - prev)
            elif( (coord_prev[0] > 0 and coord_prev[1] < 0 and coord_curr[0] < 0 and coord_curr[1] < 0) or 
                  (coord_prev[0] < 0 and coord_prev[1] < 0 and coord_curr[0] > 0 and coord_curr[1] < 0) ):
                rotation_quantity = np.pi - abs(angles_curr[i] - angles_prev[i])

            # case 4: 二三(三二)象限轉  theta = abs(cur - prev)
            elif( (coord_prev[0] < 0 and coord_prev[1] > 0 and coord_curr[0] < 0 and coord_curr[1] < 0) or 
                  (coord_prev[0] < 0 and coord_prev[1] < 0 and coord_curr[0] < 0 and coord_curr[1] > 0) ):
                rotation_quantity = abs(angles_curr[i] - angles_prev[i])

            # case 5: 一二象限轉        theta = 180度 - abs(cur - prev)
            elif( (coord_prev[0] < 0 and coord_prev[1] > 0 and coord_curr[0] > 0 and coord_curr[1] > 0) or 
                  (coord_prev[0] > 0 and coord_prev[1] > 0 and coord_curr[0] < 0 and coord_curr[1] > 0) ):
                rotation_quantity = np.pi - abs(angles_curr[i] - angles_prev[i])
            
            # ---------  跨兩象限分析(很少情況會發生): 300rpm ~ 900rpm --------- #
            # case 6: 順一三、二四轉    theta = 180度 - (cur - prev)
            # case 7: 逆三一、四二轉    theta = 180度 + (cur - prev)
            elif( (coord_prev[0] > 0 and coord_prev[1] > 0 and coord_curr[0] < 0 and coord_curr[1] < 0) or 
                  (coord_prev[0] < 0 and coord_prev[1] > 0 and coord_curr[0] > 0 and coord_curr[1] < 0) or
                  (coord_prev[0] < 0 and coord_prev[1] < 0 and coord_curr[0] > 0 and coord_curr[1] > 0) or
                  (coord_prev[0] > 0 and coord_prev[1] < 0 and coord_curr[0] < 0 and coord_curr[1] > 0) ):
                clockwise_rotation_quantity = np.pi - (angles_curr[i] - angles_prev[i])
                counterclockwise_rotation_quantity = np.pi + (angles_curr[i] - angles_prev[i])
                rotation_quantity = min(clockwise_rotation_quantity, counterclockwise_rotation_quantity)

            # case 8: 如果有點在x, y軸上，不計此點
            else:
                rotation_quantity = 0

            rotations_quantity.append(round(rotation_quantity, 5))

        ## STEP4. Take the median of the n rotation amounts and calculate the angular velocity in RPM  ##
        if len(rotations_quantity) != 0:
            avg_rotation_quantity = np.median(rotations_quantity)
        else:
            avg_rotation_quantity = 0

        angular_velocity = avg_rotation_quantity / frame_time
        spin_rate_rpm = (angular_velocity * 60) / (2 * np.pi)
        
        return spin_rate_rpm


    # Find Optical flow points and return calculated spin 
    def find_points_and_calculate_spin(self, frame, frame_no):
    
        analysis_roi = frame[0:100, 0:100]

        if len(self.old_gray) == 0 or frame_no % self.frame_interval == 1:                    # 如果還沒有上一偵資訊，就先不要算LK algorithm，先assign好old資訊
            ## Convert ROI to grayscale
            self.old_gray = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2GRAY)

            ## Key points detection: use Shi-Tomashi method Detect corners to track
            self.prev_points = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)

        else:
            ## Convert new ROI to grayscale
            new_gray = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2GRAY)

            ## Calculate optical flow using Lucas-Kanade method
            cur_points, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, new_gray, self.prev_points, None, **self.lk_params)

            ## Select good points
            if cur_points is not None:
                valid_cur_points = cur_points[st == 1]
                valid_prev_points = self.prev_points[st == 1]

                ## Draw the tracks of optical flow on the screen
                for i, (new, old) in enumerate(zip(valid_cur_points, valid_prev_points)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)


                ## Calculate Spin every 3 frames
                if(frame_no % self.frame_interval == 0):
                    new_spin_rate = self.calculate_spin_rate(valid_prev_points, valid_cur_points, 1/60)
                    self.temp_spin_rate = self.old_spin_rate

                    ## STEP6. Judge if the calculated spin rate is reasonable and smoothing the calculated RPM value ##
                    if not (np.isnan(new_spin_rate) or new_spin_rate <= 0 or new_spin_rate > 600):
                        self.old_spin_rate = int(0.8 * self.old_spin_rate + 0.2 * new_spin_rate)

                ## Update the previous frame and previous points
                self.prev_points = valid_cur_points.reshape(-1, 1, 2)
                
            self.old_gray = new_gray.copy()
            
        if frame_no % 6 == 0 or frame_no % 6 == 1 or frame_no % 6 == 2:
            return frame, self.old_spin_rate
        else:
            return frame, self.temp_spin_rate
    ### --------------------------------------------------------------------------------------------------------------- ###

    def on_frame(self, frame_result: FrameResult):
        self.track_queue.put(frame_result)

    def track_thread_funct(self, queue):
        while True:
            frame_result = queue.get()
            if frame_result is None:
                self.on_result.send(None)
                break
            logger.debug(f"Frame {frame_result.frame_id} BallTracker all done, send to on_result")
            self.on_result.send(frame_result)
        logger.debug(f"BallTracker track_thread_funct is ended")
        
    def process_frame_result(self, frame_result: FrameResult):
        if frame_result is None:
            raise ValueError("FrameResult is None")
        logger.info(f"BallTracker Frame {frame_result.frame_id} received")
        st_time = time.time()
        # Update track2ds, remove dead 2D tracks
        cam_alive_track2ds = {}
        for cam, track2ds in self.cam_track2ds.items():
            cam_alive_track2ds[cam] = {}
            for track_id, track2d in track2ds.items():
                if frame_result.frame_id - track2d.last_frame_id > 10:
                    continue
                cam_alive_track2ds[cam][track_id] = track2d
        self.cam_track2ds = cam_alive_track2ds
        timer_2d_association = measure.PerfTimer()
        timer_add_new_2d_track = measure.PerfTimer()
        # Associate 2D tracks
        for cam, detections in frame_result.camera_detections.items():
            # frame_result.camera_results[cam] = None
            track2ds_list = list(cam_alive_track2ds[cam].values())
            logger.debug(f"Frame {frame_result.frame_id} Track2d LSA listing: {len(cam_alive_track2ds[cam])} track2ds")
            logger.debug(f"Frame {frame_result.frame_id} Track2d LSA detections count: {len(detections)}")
            
            timer_2d_association.start()
            cost_matrix = np.full((len(track2ds_list), len(detections)), LARGE_NUM, dtype=np.float32)
            for track2d_idx, track2d in enumerate(track2ds_list):
                if track2d.last_frame_id < frame_result.frame_id - 10:
                    continue  # considered dead track2d
                for det_idx, (xywh, class_id, conf) in enumerate(detections):
                    cost = np.linalg.norm(track2d.xy() - np.array(xywh[:2]))
                    if cost > 150:
                        cost = LARGE_NUM
                    cost_matrix[track2d_idx, det_idx] = cost
            track_idxes, det_idxes = linear_sum_assignment(cost_matrix)
            timer_2d_association.stop()
            timer_add_new_2d_track.start()
            unmatched_detections = detections.copy()
            for track_idx, det_idx in zip(track_idxes, det_idxes):
                track2d: Track2D = track2ds_list[track_idx]
                det = detections[det_idx]
                if cost_matrix[track_idx, det_idx] > 100:
                    continue
                # if cost_matrix[track_idx, det_idx] > 50 * (frame_result.frame_id - track2d.last_frame_id):
                #     continue
                xywh, class_id, conf = det
                track2d.add_frame(frame_result.frame_id, xywh)
                unmatched_detections[det_idx] = None
            unmatched_detections = [det for det in unmatched_detections if det is not None]
            del cost_matrix
            for xywh, class_id, conf in unmatched_detections:
                new_track_id = self.camera_next_track2d_ids[cam]
                self.camera_next_track2d_ids[cam] += 1
                track2d = Track2D(cam, new_track_id)
                track2d.add_frame(frame_result.frame_id, xywh)
                cam_alive_track2ds[cam][new_track_id] = track2d
            timer_add_new_2d_track.stop()
        logger.debug(f"Frame {frame_result.frame_id} Track2d LSA 2D Association in {timer_2d_association.seconds:.2f} s (FPS: {1/max(1e-6, timer_2d_association.seconds):.2f})")
        logger.debug(f"Frame {frame_result.frame_id} Track2d LSA Add New Track in {timer_add_new_2d_track.seconds:.2f} s (FPS: {1/max(1e-6, timer_add_new_2d_track.seconds):.2f})")
        # logger.debug(f"Frame {frame_result.frame_id} Track2d LSA Total in {time.time() - st_time:.2f} s (FPS: {1/(time.time() - st_time):.2f})")
        self.cam_track2ds = cam_alive_track2ds.copy()
        frame_result.camera_track2ds = cam_alive_track2ds.copy()
        return frame_result

    def process_crossview_lsa_3d_tracking(self, frame_result: FrameResult):
        st_time = time.time()   
        cam_active_track2ds = {cam: dict() for cam in frame_result.cameras}
        cam_filtered_alive_track2ds = {cam: dict() for cam in frame_result.cameras}
        pre_check_static_cnt = 0
        logger.debug(f"Frame {frame_result.frame_id} Precheck cameras1 {frame_result.cameras}")
        logger.debug(f"Frame {frame_result.frame_id} Precheck cameras2 {frame_result.camera_track2ds}")
        for cam, track2ds in frame_result.camera_track2ds.items():
            for track_id, track2d in track2ds.items():
                if track2d.xywh(frame_result.frame_id) is None:
                    continue
                recent_xywhs = track2d.recent_xywhs(frame_result.frame_id, n=10)
                recent_xys = np.array([xywh[:2] for xywh in recent_xywhs])
                center_xy = np.mean(recent_xys, axis=0) if len(recent_xys) > 0 else None
                if (center_xy is not None and (max_center_dist:=np.linalg.norm(recent_xys - center_xy, axis=1).max()) < 5):
                    pre_check_static_cnt += 1
                    logger.debug(f"Frame {frame_result.frame_id} precheck False Track2d {cam.name}-{track2d.track_id} considered static (len={len(recent_xys)}, center_xy={center_xy}, maxdist={max_center_dist}, dists={np.linalg.norm(recent_xys - center_xy, axis=1)}")
                    cam_filtered_alive_track2ds[cam][track_id] = track2d                        
                else:
                    logger.debug(f"Frame {frame_result.frame_id} precheck OK Track2d {cam.name}-{track2d.track_id} considered non-staic")
                    cam_active_track2ds[cam][track_id] = track2d
        frame_result.camera_active_track2ds = cam_active_track2ds.copy()
        frame_result.camera_filtered_track2ds = cam_filtered_alive_track2ds.copy()

        cam1, cam2 = frame_result.cameras[0], frame_result.cameras[1]
        # Use only 2 cameras for now
        cameraset = self.cameraset
        cam1_tracks = list(cam_active_track2ds[cam1].values())
        cam2_tracks = list(cam_active_track2ds[cam2].values())
        cost_matrix = np.full((len(cam1_tracks), len(cam2_tracks)), 999999, dtype=np.float32)
        logger.info(f"Frame {frame_result.frame_id} Using cameras {cam1.name}({len(cam1_tracks)}) and {cam2.name}({len(cam2_tracks)}) for tracking")
        epiline_cache = {}
        c1c2_product = list(product(range(len(cam1_tracks)), range(len(cam2_tracks))))
        timer_epiline = measure.PerfTimer()
        logger.debug(f"Frame {frame_result.frame_id} Cross-View Track2d Product: {len(c1c2_product)}")
        for track1_idx, track2_idx in c1c2_product:
            track1, track2 = cam1_tracks[track1_idx], cam2_tracks[track2_idx]
            # calculate cost
            common_frame_ids = set(track1.frame_id_xywh.keys()) & set(track2.frame_id_xywh.keys())
            common_frame_ids = sorted(list(common_frame_ids))[-min(5, len(common_frame_ids)):]
            logger.debug(f"{cam1.name}:{track1.track_id} and {cam2.name}:{track2.track_id} Common frame ids: {common_frame_ids}")
            if len(common_frame_ids) < 3:
                continue
            F = cameraset.get_fmat(cam1, cam2)
            cost = 0
            for frame_id in common_frame_ids:
                xywh1 = track1.frame_id_xywh[frame_id]
                xywh2 = track2.frame_id_xywh[frame_id]
                pt1 = np.array(xywh1[:2])
                pt2 = np.array(xywh2[:2])
                timer_epiline.start()
                line = cameraset.get_epiline(cam1, cam2, pt1, F, whichImage=1)
                if cam2 not in frame_result.camera_epilines:
                    frame_result.camera_epilines[cam2] = {}
                if frame_id == frame_result.frame_id:
                    frame_result.camera_epilines[cam2][track1] = line
                frame_cost = np.abs(line @ np.array([*pt2, 1])) / np.linalg.norm(line[:2])
                timer_epiline.stop()
                if frame_cost > 30:
                    cost = 999999
                    break
                cost += frame_cost
            cost_avg = cost / len(common_frame_ids)
            if cost >= 999999 or cost_avg > 30:
                cost = 999999
            else:
                cost = cost_avg
            cost_matrix[track1_idx, track2_idx] = cost
        logger.debug(f"Frame {frame_result.frame_id} Cross-View Cost matrix:({cost_matrix.shape}) \n{cost_matrix}")
        if timer_epiline.seconds > 1e-6:
            logger.debug(f"Frame {frame_result.frame_id} Cross-View Epiline in {timer_epiline.seconds:.2f}s (FPS: {1/timer_epiline.seconds:.2f})")
        track1_idxes, track2_idxes = linear_sum_assignment(cost_matrix)
        track_pairs = []
        for track1_idx, track2_idx in zip(track1_idxes, track2_idxes):
            if cost_matrix[track1_idx, track2_idx] > 30:
                continue
            track1 = cam1_tracks[track1_idx]
            track2 = cam2_tracks[track2_idx]
            track_pairs.append((track1, track2))
        del cost_matrix
        logger.debug(f"Frame {frame_result.frame_id} Cross-View Track pairs: {len(track_pairs)}")

        # triangulate 3D position
        track_pair_pos3ds = []
        for track1, track2 in track_pairs:
            pt1, pt2 = track1.xy(frame_result.frame_id), track2.xy(frame_result.frame_id)
            points4d = cv.triangulatePoints(cam1.projection, cam2.projection, pt1, pt2)
            points3d = (points4d[:3, :]/points4d[3, :]).T
            pos3d = points3d[0]
            track_pair_pos3ds.append((track1, track2, pos3d))
            logger.debug(f"Triangulated 3D position: [{', '.join([f'{p:.2f}' for p in pos3d])}]")

        # Associate 3D tracks
        alive_track3ds_list = []
        for track3d in self.track3ds_list:
            if (frame_result.frame_id - track3d.last_frame_id) < 10:
                alive_track3ds_list.append(track3d)
        
        cost_matrix = np.full((len(alive_track3ds_list), len(track_pair_pos3ds)), LARGE_NUM, dtype=np.float32)
        for pair_idx, (track1, track2, pos3d) in enumerate(track_pair_pos3ds):
            for track3d_idx, track3d in enumerate(alive_track3ds_list):
                track3d_pos3d = track3d.recent_pos3ds(frame_result.frame_id, n=10)
                if len(track3d_pos3d) == 0:
                    continue
                track3d_pos3d = track3d_pos3d[0]
                cost = np.linalg.norm(track3d_pos3d - pos3d)
                if cost > 2.0:
                    cost = LARGE_NUM
                cost_matrix[track3d_idx, pair_idx] = cost
        logger.info(f"Frame {frame_result.frame_id} Track3d-Pairs Cost matrix: ({cost_matrix.shape})\n{cost_matrix}")
        track3d_idxes, track_pair_idxes = linear_sum_assignment(cost_matrix)
        unmatched_track_pairs = track_pair_pos3ds.copy()
        for track3d_idx, track_pair_idx in zip(track3d_idxes, track_pair_idxes):
            if cost_matrix[track3d_idx, track_pair_idx] > 2.0:
                continue
            track3d: Track3D = alive_track3ds_list[track3d_idx]
            ttp = track_pair_pos3ds[track_pair_idx]
            track1, track2, pos3d = ttp
            track3d.add_frame(frame_result.frame_id, pos3d, {cam1: track1, cam2: track2})
            unmatched_track_pairs.remove(ttp)
        del cost_matrix
        for track1, track2, pos3d in unmatched_track_pairs:
            track3d = Track3D(self.next_track3d_id)
            track3d.add_frame(frame_result.frame_id, pos3d, {cam1: track1, cam2: track2})
            alive_track3ds_list.append(track3d)
            self.next_track3d_id += 1
        self.track3ds_list = alive_track3ds_list.copy()
        frame_result.track3ds_list = self.track3ds_list.copy()
        # logger.info(f"Frame {frame_result.frame_id} Triangulate and LSA Alive Track3ds: {len(alive_track3ds_list)} in {time.time() - st_time:.2f}s (FPS: {1/(time.time() - st_time):.2f})")
        st_time = time.time()
        frame_result = self.process_select_main(frame_result)
        # logger.info(f"Frame {frame_result.frame_id} Select main and Measure {len(alive_track3ds_list)} in {time.time() - st_time:.2f}s (FPS: {1/(time.time() - st_time):.2f})")
        return frame_result
    
    def process_select_main(self, frame_result: FrameResult):
        st_time = time.time()
        # select main ball
        track3d_list = frame_result.track3ds_list
        if len(track3d_list) > 0:
            alive_track3ds_list = []
            for track3d in track3d_list:
                if track3d.pos3d(frame_result.frame_id) is None:
                    continue
                alive_track3ds_list.append(track3d)
            logger.info(f"Frame {frame_result.frame_id} Selecting main ball from {len(alive_track3ds_list)} track3ds")
            main_track3d = None
            main_track3d_candidates = []
            MIN_SAMPLE_FRAME_COUNT = 6
            for track3d in alive_track3ds_list:
                # check #1 : track3d can't be static
                if len(track3d.frame_id_pos3ds) < MIN_SAMPLE_FRAME_COUNT:
                    logger.debug(f"Track3d {track3d.track_id} not enough frames")
                    continue
                if not geometry.is_in_court(track3d.pos3d(frame_result.frame_id)):
                    logger.debug(f"Track3d {track3d.track_id} not in court")
                    continue
                prev_pos3ds = track3d.recent_pos3ds(frame_result.frame_id, n=30)
                if len(prev_pos3ds) < MIN_SAMPLE_FRAME_COUNT:
                    logger.debug(f"Track3d {track3d.track_id} not enough pos3ds")
                    continue
                prev_pos3ds = prev_pos3ds[:min(30, len(prev_pos3ds)):max(1, len(prev_pos3ds) // 6)]
                shift_pos3ds = np.diff(prev_pos3ds, axis=0)
                logger.debug(f"Track3d {track3d.track_id} 3D shift pos3ds: {shift_pos3ds}")
                shift_norms = np.linalg.norm(shift_pos3ds, axis=1)
                logger.debug(f"Track3d {track3d.track_id} 3D shift norms: {shift_norms}, mean: {np.mean(shift_norms)}")
                if np.mean(shift_norms) < 0.05:
                    logger.debug(f"Track3d {track3d.track_id} considered 3D static")
                    continue
                total_cams = 0
                valid_cams = 0
                # check #2 : track2ds can't be ALL static, 
                # this can effectively filter out the static balls that cause false moving 3D positions.
                for cam, track2d in track3d.cam_track2ds(frame_result.frame_id).items():
                    total_cams += 1
                    prev_xywhs = track2d.recent_xywhs(frame_result.frame_id, n=30)
                    if len(prev_xywhs) < 6:
                        continue
                    prev_xy = np.array([xywh[:2] for xywh in prev_xywhs])
                    prev_xy = prev_xy[::max(1, len(prev_xy) // 5)]
                    shift_xy = np.diff(prev_xy, axis=0)
                    shift_norms = np.linalg.norm(shift_xy, axis=1)
                    logger.debug(f"Track3d {track3d.track_id} {track3d.pos3d(frame_result.frame_id)} 2D shift norms mean: {np.mean(shift_norms)}")
                    if np.mean(shift_norms) < 10:
                        logger.debug(f"Track2d {track2d.track_id} considered 2D static in cam {cam.name}")
                        continue
                    valid_cams += 1
                if valid_cams == 0:
                    logger.debug(f"Track3d {track3d.track_id} all cam static, skipped")
                    continue
                main_track3d_candidates.append(track3d)

            if len(main_track3d_candidates) > 0:
                court_center = np.array([4.5, 9.0, 0.0])
                main_track3d = main_track3d_candidates[0]
                main_dist = np.linalg.norm(main_track3d.pos3d(frame_result.frame_id) - court_center)
                for track3d in main_track3d_candidates:
                    dist = np.linalg.norm(track3d.pos3d(frame_result.frame_id) - court_center)
                    if dist < main_dist:
                        main_dist = dist
                        main_track3d = track3d
                logger.info(f"Frame {frame_result.frame_id} Main ball selected: {main_track3d.track_id}")

            if main_track3d is not None:
                frame_result.main_track3d = main_track3d
                frame_result.main_pos3d = main_track3d.pos3d(frame_result.frame_id)
            logger.info(f"Frame {frame_result.frame_id} main ball pos3d: {frame_result.main_pos3d}")

        st_time = time.time()
        # calculate velocity, acceleration
        main_pos3d = frame_result.main_pos3d
        if main_pos3d is not None:
            ball_pos, ball_pos_frame_id = main_pos3d, frame_result.frame_id
            ball_vel, ball_vel_frame_id = self.ball_pos_var.input(ball_pos_frame_id, ball_pos)
            ball_acc, ball_acc_frame_id = self.ball_vel_var.input(ball_vel_frame_id, ball_vel)
            ball_jer, ball_jer_frame_id = self.ball_acc_var.input(ball_acc_frame_id, ball_acc)
            logger.info(f"Frame {frame_result.frame_id} ball pos: {format_array(main_pos3d)} vel: {format_array(ball_vel)}, acc: {format_array(ball_acc)}, jer: {format_array(ball_jer)}")
            vel_kmh = np.linalg.norm(ball_vel) / 1000 * (self.fps * 3600) if ball_vel is not None else None
            acc_ms2 = np.linalg.norm(ball_acc) * (self.fps * self.fps) if ball_acc is not None else None
            jer_ms3 = np.linalg.norm(ball_jer) * (self.fps * self.fps * self.fps) if ball_jer is not None else None
            frame_result.ball_vel = ball_vel
            frame_result.ball_acc = ball_acc
            frame_result.ball_jer = ball_jer
            frame_result.ball_vel_kmh = vel_kmh
            frame_result.ball_acc_ms2 = acc_ms2
            frame_result.ball_jer_ms3 = jer_ms3
            
            ### ----------------------------------------------- New Add ------------------------------------------------------- ###
            ### 目前要想一下這邊要怎麼拿到2D bounding box, 在做roiHandling, 再能傳入下面find_points_and_calculate_spin()裡面
            # 目前先用frame_result main_track3d試看看, 來回推找camera A的2d bounding box
            cam_0 = frame_result.cameras[0]
            camA_main_ball_2DBoundingBox = (frame_result.main_track3d.cam_track2ds(frame_result.frame_id))[cam_0]
            x, y, w, h = camA_main_ball_2DBoundingBox.xywh(frame_result.frame_id)
            # print(x, y, w, h)

            frame = frame_result.camera_result_frames[cam_0]
            
            ## Define the target ball's roi
            roiHandler = roiPreprocesser.RoiHandler()
            roi = roiHandler.find_roi(frame, x, y, w, h)
            upscaled_roi = roiHandler.enhance_image(roi)
            roiHandler.set_xyoffset(0, 0)
            frame[roiHandler.y_offset:roiHandler.y_offset+upscaled_roi.shape[0], roiHandler.x_offset:roiHandler.x_offset+upscaled_roi.shape[1]] = upscaled_roi

            ## Calculate spin rate: using Optical flow 
            frame, frame_result.spin_rate_rpm = self.find_points_and_calculate_spin(frame, frame_result.frame_id)
            print(f"SPIN: {frame_result.spin_rate_rpm}")
            
            ## Draw bounding box and spin rate
            x1, y1, x2, y2 = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            
            ## Setting the visualization with green to red
            gamma = (frame_result.spin_rate_rpm - 0) / 300          # if now_spin_rate > 300rpm -> so fast, set to red 
            g_value = int(round(255 * (1 - gamma)))
            r_value = int(round(255 * gamma))

            # 保證顏色值在0到255之間
            g_value = max(0, min(255, g_value))
            r_value = max(0, min(255, r_value))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, g_value, r_value), 3)
            cv2.putText(frame, f"Spin: {frame_result.spin_rate_rpm}rpm", (125, 75), cv2.FONT_HERSHEY_TRIPLEX, 2.5, (0, g_value, r_value), 5)
            frame_result.camera_result_frames[cam_0] = frame
            ### --------------------------------------------------------------------------------------------------------------- ###

            tilt_angle_deg = geometry.vec_tilt_angle_deg(frame_result.ball_vel) if frame_result.ball_vel is not None else None
            yaxis_angle_deg = geometry.vec_yaxis_angle_deg(frame_result.ball_vel) if frame_result.ball_vel is not None else None
            frame_result.ball_tilt_angle_deg = tilt_angle_deg
            frame_result.ball_yaxis_angle_deg = yaxis_angle_deg
            logger.info(f"Frame {frame_result.frame_id} ball vel_kmh : {vel_kmh} acc_ms2: {acc_ms2} jer_ms3: {jer_ms3}")
        return frame_result


# Detect rally from BallTracker's result 
class RallyState(enum.Enum):
    WAIT_SERVE = enum.auto()
    WAIT_SERVE_RECEIVE = enum.auto()
    RALLY = enum.auto()
    RALLY_RESET = enum.auto()
    COOLDOWN = enum.auto()
    # There is actually a WAIT_CROSS_NET state, it should be added to make the function stateless.

class RallyTracker:
    on_result = Signal(FrameResult)
    def __init__(self, fps=60, is_video_mode=True):
        self.buffer_frames = []
        self.buffer_len = 120
        self.fps = fps
        self.is_video_mode = is_video_mode  # True when using manual cutted video, False to enable auto detect

        self.ball_lost_time = 0
        self.ball_lost_time_threshold = fps * 2
        self.cooldown_time = 0
        self.cooldown_time_threshold = fps * 1
        self.wait_receive_time = 0
        self.wait_receive_time_threshold = fps * 5

        self.state = RallyState.WAIT_SERVE
        self.recent_collide_frame = None
        self.recent_attack_frame = None
        self.served = False
        self.serve_cross_net = False
        self.serve_frame = None
        self.serve_collide_frame = None  # for better cross net timing
        self.serve_frame_id_pos3d = {}
        self.serve_receive_frame = None
        self.serve_cross_net_frame = None

        self.prepend_length = 120
        self.prepend_frame_results = []

    def init_rally(self):
        self.recent_collide_frame = None
        self.recent_attack_frame = None
        self.served = False
        self.serve_cross_net = False
        self.serve_frame = None
        self.serve_collide_frame = None
        self.serve_frame_id_pos3d = {}
        self.serve_receive_frame = None
        self.serve_cross_net_frame = None
        self.ball_lost_time = 0
        self.wait_receive_time = 0

        self.prepend_frame_results = []
    
    def on_frame(self, frame_result: FrameResult):
        if frame_result is None:  # Sentinel
            self.buffer_frames = []
            self.on_result.send(None)
            return
        st_time = time.time()
        self.buffer_frames.append(frame_result)
        if len(self.buffer_frames) > self.buffer_len:
            self.buffer_frames.pop(0)

        # Detect collision
        frame_id = frame_result.frame_id
        rcf = self.recent_collide_frame
            # frame_result.ball_jer_ms3 > 4200 and \
            # frame_result.ball_jer_ms3 > 3600 and \
        if frame_result.ball_jer_ms3 is not None and \
            frame_result.ball_jer_ms3 > 3400 and \
            (rcf is None or (frame_id - rcf.frame_id > 20)):
            frame_result.event_is_collide = True
            self.recent_collide_frame = frame_result
            logger.info(f"Frame {frame_id} collision detected")

        frame_result.rally_tracker_state = self.state
        next_state = self.state
        if self.state == RallyState.WAIT_SERVE:
            self.prepend_frame_results.append(frame_result)
            if len(self.prepend_frame_results) > self.prepend_length:
                fr = self.prepend_frame_results.pop(0)
                self.on_result.send(fr)
                # time.sleep((1/60) * 0.8) # TVL20 hotfix

            # if rcf is not None and 2 <= frame_id - rcf.frame_id <= 20 and \
            sample_buffer = self.buffer_frames[-min(8, len(self.buffer_frames)):]
            sample_frame_result = sample_buffer[0]
            if sample_frame_result.ball_vel is not None and sample_frame_result.main_pos3d is not None and \
                geometry.is_in_court(sample_frame_result.main_pos3d):
                ball_vel_kmh = sample_frame_result.ball_vel_kmh
                tilt = sample_frame_result.ball_tilt_angle_deg
                yaxis_angle = sample_frame_result.ball_yaxis_angle_deg
                logger.info(f"Frame {frame_id} ball vel: {format_array(sample_frame_result.ball_vel)} vel_kmh: {ball_vel_kmh} tilt: {tilt} yaxis_angle: {yaxis_angle}")
                if 30 <= ball_vel_kmh < 150 and -15 <= tilt <= 45 and yaxis_angle <= 30:
                    # nested 'if' is bad, but it's ok for now
                    self.served = True
                    self.serve_frame = sample_frame_result
                    self.serve_frame_id_pos3d[frame_id] = sample_frame_result.main_pos3d
                    frame_result.event_is_serve = True
                    # Find collide before
                    if self.recent_collide_frame is not None and (frame_id - self.recent_collide_frame.frame_id) <= 50:
                        self.serve_collide_frame = self.recent_collide_frame
                    else:
                        self.serve_collide_frame = self.serve_frame  # use serve frame as collide frame if not found
                    # Find max speed
                    max_vel_kmh = max([fr.ball_vel_kmh for fr in sample_buffer if fr.ball_vel_kmh is not None])
                    frame_result.serve_speed_kmh = max_vel_kmh
                    # WARNING! serve_start_pos3d should be put when receive for convienence 
                    # (for rendering trajectory to take start/end points at the same time)
                    # frame_result.serve_start_pos3d = self.serve_frame.main_pos3d
                    next_state = RallyState.WAIT_SERVE_RECEIVE
                    # frame_result.event_is_rally_begin = True
                    # frame_result.event_is_rally = True
                    self.prepend_frame_results[0].event_is_rally_begin = True
                    for fr in self.prepend_frame_results:
                        fr: FrameResult
                        fr.event_is_rally = True
                        self.on_result.send(fr)
                    self.prepend_frame_results = []
                    
                    logger.info(f"Frame {frame_id} serve detected {max_vel_kmh}")

        elif self.state == RallyState.WAIT_SERVE_RECEIVE:
            frame_result.event_is_rally = True
            self.serve_frame_id_pos3d[frame_id] = frame_result.main_pos3d
            self.wait_receive_time += 1
            # Force quit if wait receive for too long.
            if self.wait_receive_time > self.wait_receive_time_threshold:
                self.wait_receive_time = 0
                next_state = RallyState.RALLY

            if not self.serve_cross_net:
                if frame_result.main_pos3d is not None and self.serve_frame.main_pos3d is not None \
                    and ((frame_result.main_pos3d[1] > 9) ^ (self.serve_frame.main_pos3d[1] > 9)):
                    self.serve_cross_net = True
                    self.serve_cross_net_frame = frame_result
                    frame_result.event_is_serve_cross_net = True
                    frame_result.serve_cross_net_seconds = (frame_result.frame_id - self.serve_collide_frame.frame_id) / self.fps
                    logger.info(f"Frame {frame_id} serve cross net detected")
            else: # serve crossed net
                if frame_result.event_is_collide \
                    or ((frame_result.frame_id - self.serve_cross_net_frame.frame_id > 2) and \
                        (frame_result.ball_vel is not None and frame_result.ball_vel @ np.array([0, 0, 1]) * 3.6 > 10)):
                    self.serve_receive_frame = frame_result
                    # estimate the landing point
                    serve_pos3d_list = {fid: pos3d for fid, pos3d in self.serve_frame_id_pos3d.items() if pos3d is not None}
                    if len(serve_pos3d_list) > 10:
                        # get latest 10 frames
                        fids = sorted(list(serve_pos3d_list.keys()))[-min(12, len(serve_pos3d_list)):-2]
                        serve_pos3d_list = {fid: serve_pos3d_list[fid] for fid in fids}
                        logger.info(f"Serve trajectory: {serve_pos3d_list}")
                        fit_st_time = time.time()
                        trajectory = geometry.fit_trajectory(serve_pos3d_list)
                        landing_pos = geometry.get_trajectory_land_pos(trajectory)
                        frame_result.serve_start_pos3d = self.serve_frame.main_pos3d
                        frame_result.serve_end_pos3d = landing_pos
                        frame_result.event_is_serve_receive = True
                        logger.info(f"Serve trajectory landing pos: {format_array(landing_pos)}")
                        # with open('fit_time.txt') as fw:
                        #     fw.write(f"{time.time() - fit_st_time:.2f} (FPS {1/(time.time() - fit_st_time):.2f})\n")
                    else:
                        logger.warning(f"Serve trajectory not enough points, which should not happended.")
                    next_state = RallyState.RALLY
            self.on_result.send(frame_result)

        elif self.state == RallyState.RALLY:
            # Detect Attack
            sample_buffer = self.buffer_frames[-min(6, len(self.buffer_frames)):]
            sample_frame_result = sample_buffer[0]
                # and self.recent_collide_frame is not None and 6 <= self.recent_collide_frame.main_pos3d[1] <= 12 \
            if frame_result.main_pos3d is not None and frame_result.ball_vel_kmh is not None \
                    and (self.recent_attack_frame is None or (frame_id - self.recent_attack_frame.frame_id) > 60) \
                    and self.recent_collide_frame is not None and 4 <= self.recent_collide_frame.main_pos3d[1] <= 14 \
                    and (2 < (frame_id - self.recent_collide_frame.frame_id) < 6) \
                    and frame_result.ball_vel_kmh >= 40 \
                    and frame_result.ball_tilt_angle_deg is not None and frame_result.ball_tilt_angle_deg <= 30:
                # find max speed
                max_vel_kmh = max([fr.ball_vel_kmh for fr in sample_buffer if fr.ball_vel_kmh is not None])
                if max_vel_kmh >= 45:
                    frame_result.event_is_spike = True
                    frame_result.spike_speed_kmh = max_vel_kmh
                    frame_result.spike_height_m = frame_result.main_pos3d[2]
                    logger.info(f"Frame {frame_id} spike detected {max_vel_kmh}")
                else:
                    frame_result.event_is_attack = True
                    frame_result.attack_speed_kmh = frame_result.ball_vel_kmh
                    frame_result.attack_height_m = frame_result.main_pos3d[2]
                    logger.info(f"Frame {frame_id} attack detected {frame_result.ball_vel_kmh}")
                self.recent_attack_frame = frame_result

            # detect the end of rally...
            frame_result.event_is_rally = True
            if frame_result.main_pos3d is None:
                self.ball_lost_time += 1
                if self.ball_lost_time > self.ball_lost_time_threshold:
                    self.cooldown_time = 0
                    if not self.is_video_mode:
                        next_state = RallyState.COOLDOWN
                        frame_result.event_is_rally_end = True
            else:
                self.ball_lost_time = 0
            self.on_result.send(frame_result)

        elif self.state == RallyState.COOLDOWN: # wait some time before next rally...
            if self.cooldown_time > self.cooldown_time_threshold:
                next_state = RallyState.RALLY_RESET
            self.cooldown_time += 1
            self.on_result.send(frame_result)

        elif self.state == RallyState.RALLY_RESET:
            self.init_rally()
            frame_result.event_is_rally_reset = True
            next_state = RallyState.WAIT_SERVE
            self.on_result.send(frame_result)

        # frame_result.rally_tracker_state = self.state
        # logger.info(f"Frame {frame_id} rally state: {self.state}, RallyTracker FPS: {1/(time.time() - st_time):.2f}")

        # self.on_result.send(frame_result)  # send the frame_result whatever the state is, because demo needs it.

        if self.state != next_state:
            logger.info(f"Frame {frame_id} state change: {self.state} -> {next_state}")
        self.state = next_state


class RallyOutputManager:
    # This class is for setting the output path for camera outputs, json,... etc.
    # The file IO is not handled here, it's left for camera write thread for better management,
    # because separating the create and release in different class is hard to control.
    on_result = Signal(FrameResult)
    def __init__(self, output_dir, camera_inputs, is_video_mode=True) -> None:
        self.output_dir = output_dir
        self.is_video_mode = is_video_mode
        self.camera_output_paths = {}
        self.rally_dir = None
        if is_video_mode:
            self.rally_dir = output_dir
            for cam, input_path in camera_inputs.items():
                output_path = Path(output_dir) / f"{Path(input_path).name}_detect.mp4"
                self.camera_output_paths[cam] = output_path
                logger.info(f"Video Mode: Camera {cam.name} output paths: {output_path}")

    def on_frame(self, frame_result: FrameResult):
        if frame_result is None:
            self.on_result.send(None)
            return

        if self.is_video_mode:
            # Remain the init setting
            # frame_result.camera_output_paths = self.camera_output_paths
            frame_result.rally_dir = self.output_dir
            pass
        else:
            if frame_result.event_is_rally_begin:
                elapsed_time = frame_result.frame_id / frame_result.fps
                rally_start_time = elapsed_time
                rally_dirname = get_rally_dir_from_datetime(frame_result.start_time, elapsed_time)
                self.rally_dir = self.output_dir / rally_dirname
                self.rally_dir.mkdir(parents=True, exist_ok=True)
                for cam in frame_result.cameras:
                    rally_filename = get_rally_video_name_from_datetime(frame_result.start_time, elapsed_time, cam.name)
                    # output_path = (self.output_dir / rally_filename).with_suffix(".mp4")
                    output_path = (self.rally_dir / rally_filename).with_suffix(".mp4")
                    self.camera_output_paths[cam] = output_path
                    logger.info(f"Auto Mode: Camera {cam.name} output path: {output_path}")
            if frame_result.event_is_rally_reset:  # TVL20 hotfix
                # self.rally_dir = None
                # self.camera_output_paths = {}
                pass
        frame_result.camera_output_paths = self.camera_output_paths
        frame_result.rally_dir = self.rally_dir
        self.on_result.send(frame_result)


class ResultRenderer:
    on_result = Signal(FrameResult)
    def __init__(self, cameras, video_size=(1920, 1080), fps=60, is_video_mode=True, is_sponsor_mode=False) -> None:
        # self.cmap_track_id = matplotlib.cm.get_cmap('hsv', 20)
        # self.cmap_3d = matplotlib.cm.get_cmap('hsv', 20)
        self.cmap_track_id = matplotlib.pyplot.get_cmap('hsv', 20)
        self.cmap_3d = matplotlib.pyplot.get_cmap('hsv', 20)
        self.is_sponsor_mode = is_sponsor_mode
        if is_sponsor_mode:
            self.sponsors_image = cv.imread("sponsors.png", cv.IMREAD_UNCHANGED) if Path("sponsors.png").exists() else None
        else:
            self.sponsors_image = None
        self.collide_count = 0
        self.cmap = matplotlib.pyplot.get_cmap('hsv', 20)
        self.fps = fps
        self.cameras = cameras
        self.video_size = video_size
        self.is_video_mode = is_video_mode
        self.camera_output_paths = {}  # for recording the change of output path, so we can change out.
        self.output_path_outs = {}
        self.camera_render_threads = {}
        self.camera_render_queues = {}
        self.camera_write_threads = {}
        self.camera_write_queues = {}
        self.camera_render_done_syncers = {}
        self.camera_render_finish_syncer = Queue()
        self.camera_prepend_buffers = {}

        self.camera_render_info_card_send_stages = {}
        self.camera_render_info_card_receive_stages = {}  # put the result frame from processes back into frame_result
        self.camera_render_info_card_process_stages = {}
        self.camera_render_3d_court_stages = {}
        # First, because the multi-processing stages can only receive raw data,
        # so this queue is for passing the frame_result to the stage after multi-processing
        # Second, the data and frame need to be separated for frame share memory on numpy array.
        self.camera_render_info_cards_data_queue = {}
        self.camera_render_info_cards_input_frame_queue = {}
        self.camera_render_info_cards_output_frame_queue = {}
        self.camera_sponsor_image_queues = {}
        self.camera_court_plane_image_queues = {}
        self.camera_court_3d_image_queues = {}

        self.prepend_length = fps * 3
        self.frame_result_queue = Queue(maxsize=60*3)  # this is increased to >120 because the prepend is now handle by RallyTracker
        self.court_3d_figure = visuals.Court3DFigure()
        self.court_plane_figure = visuals.Court2DFigure(image_size=(300,600))
        self.rate_limiter = RateLimiter(1 / (fps * 1.4)) # TVL20 hotfix
        
        for cam in cameras:
            # self.camera_outs[cam] = out = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*'mp4v'), fps, video_size)
            self.camera_render_queues[cam] = render_q = Queue(maxsize=3)
            self.camera_write_queues[cam] = write_q = Queue(maxsize=5 * self.fps)
            self.camera_sponsor_image_queues[cam] = sponsor_queue = ArrayQueue(1000) if self.sponsors_image is not None else None
            self.camera_court_plane_image_queues[cam] = court_plane_queue = ArrayQueue(1000)
            self.camera_court_3d_image_queues[cam] = court_3d_queue = ArrayQueue(1000)
            self.camera_prepend_buffers[cam] = []
            self.camera_render_info_cards_data_queue[cam] = data_queue = multiprocessing.Queue(maxsize=3)
            self.camera_render_info_cards_input_frame_queue[cam] = input_frame_queue = ArrayQueue(1000)  # shared memory queue for numpy arrays
            self.camera_render_info_cards_output_frame_queue[cam] = output_frame_queue = ArrayQueue(1000)  # shared memory queue for numpy arrays
            self.camera_render_info_card_send_stages[cam] = rinfo_send_stage = PipelineStage(partial(self.render_infocards_stage_send_funct, cam, input_frame_queue, court_plane_queue, court_3d_queue, sponsor_queue, data_queue), input_queue=render_q)
            self.camera_render_info_card_process_stages[cam] = rinfo_process_stage = ProcessPipelineStage(partial(self.render_infocards_stage_process_funct, input_frame_queue, court_plane_queue, court_3d_queue, sponsor_queue, output_frame_queue), \
                                                                                                          input_queue=data_queue, output_queue=NullQueue(), process_name=f"rinfo_process_stage-{cam.name}")
            self.camera_render_info_card_receive_stages[cam] = rinfo_receive_stage = PipelineStage(partial(self.render_infocards_stage_receive_funct, cam, output_frame_queue), \
                                                                                                   input_queue=rinfo_send_stage.output_queue, on_finished=partial(lambda q: q.put(None), data_queue))
            self.camera_render_3d_court_stages[cam] = r3d_stage = PipelineStage(partial(self.render_3d_court, cam), input_queue=rinfo_receive_stage.output_queue)

            self.camera_render_threads[cam] = Thread(target=self.camera_render_thread_funct, args=(cam, r3d_stage.output_queue, write_q))
            self.camera_render_threads[cam].start()
            self.camera_write_threads[cam] = Thread(target=self.camera_write_thread_funct, args=(cam, write_q))
            self.camera_write_threads[cam].start()

        self.render_3d_court_data_queue = multiprocessing.Queue(maxsize=3)
        self.render_3d_court_rendered_queue = ArrayQueue(300)
        self.render_3d_court_send_stage = PipelineStage(partial(self.render_3d_court_send_funct, self.render_3d_court_data_queue), \
                                                        on_finished=partial(lambda: self.render_3d_court_data_queue.put(None)))
        self.render_3d_court_process = multiprocessing.Process(target=self.render_3d_court_process_funct, \
                                                                     args=(self.render_3d_court_data_queue, self.render_3d_court_rendered_queue))
        self.render_3d_court_receive_stage = PipelineStage(partial(self.render_3d_court_receive_funct, self.render_3d_court_rendered_queue), \
                                                           input_queue=self.render_3d_court_send_stage.output_queue, output_queue=NullQueue(),
                                                           on_finished=partial(lambda: self.stop_camera_render_threads()))
        # t = Thread(target=self.render_thread_funct, args=(self.frame_result_queue,))
        t = Thread(target=self.render_thread_funct, args=(self.frame_result_queue, self.render_3d_court_send_stage.input_queue))
        self.render_3d_court_process.start()
        t.start()
        self.write_thread = t
        self.serve_speed_kmh = None
        self.serve_cross_net_seconds = None
        self.spike_speed_kmh = None
        self.spike_height_m = None

        self.cache_court_3d_figure = None
        self.cache_court_plane_image = None

    def join(self):
        self.write_thread.join()
        for cam_thread in self.camera_write_threads.values():
            cam_thread.join()

    def reset_rally(self):
        self.court_3d_figure = visuals.Court3DFigure()
        self.court_plane_figure = visuals.Court2DFigure(image_size=(300,600))
        self.collide_count = 0
        self.cache_court_3d_figure = None
        self.cache_court_plane_image = None

    def on_frame(self, frame_result: FrameResult):
        if not self.write_thread.is_alive():
            logger.warning("Write thread is dead, but still giving frame result")
        # self.rate_limiter.wait()  # TVL20 hotfix
        self.frame_result_queue.put(frame_result)
        frame_id = frame_result.frame_id if frame_result is not None else None
        if self.frame_result_queue.full():
            logger.warning("Frame result queue is full, frame_id: {}", frame_id)

    def render_thread_funct(self, frame_result_queue, output_queue):
        # Render the frame-level result here,
        # left the camera-level result later
        while True:
            self.rate_limiter.wait()  # TVL20 hotfix
            frame_result: FrameResult = frame_result_queue.get()
            st_time = time.time()
            if frame_result is None:
                output_queue.put(None)
                logger.debug(f"Write thread received None, stop writing")
                break
            frame_id = frame_result.frame_id
            if frame_id % 300 == 0:
                gc_st_time = time.time()
                # collected = gc.collect()
                collected = 0
                logger.info(f"Frame {frame_id} GC collected: {collected} in {time.time() - gc_st_time:.4f} s")
            if frame_result.event_is_rally_reset:
                self.reset_rally()

            st_time = time.time()
            if frame_result.event_is_serve_receive and \
                frame_result.serve_start_pos3d is not None and frame_result.serve_end_pos3d is not None:
                self.court_plane_figure.add_serve_pos3d(frame_result.serve_start_pos3d, frame_result.serve_end_pos3d)
                pass
            if True:
                self.cache_court_plane_image = self.court_plane_figure.get_court_image()
            frame_result.court_plane_image = self.cache_court_plane_image 
            # logger.info(f"Frame {frame_result.frame_id} court plane images rendered in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f})")

            output_queue.put(frame_result)
        for cam_thread in self.camera_render_threads.values():
            cam_thread.join()

    def render_3d_court_send_funct(self, data_queue, frame_result: FrameResult):
        is_rally_end = frame_result.event_is_rally_end
        is_rally = frame_result.event_is_rally
        frame_id = frame_result.frame_id
        is_collide = frame_result.event_is_collide
        main_pos3d = frame_result.main_pos3d
        cmap = self.cmap
        if is_rally_end:
            self.collide_count = 0
        if is_collide:
            self.collide_count += 1
        color = cmap((self.collide_count * 3) % 20) if main_pos3d is not None else None
        frame_result.main_color = color
        data_queue.put((is_rally_end, is_collide, main_pos3d, color, is_rally, frame_id))
        logger.debug(f"Frame {frame_result.frame_id} court 3d images sent to render_3d_court_process_funct")
        return frame_result
    
    @staticmethod
    def render_3d_court_process_funct(data_queue, rendered_queue):
        c3f = visuals.Court3DFigure()
        while True:
            data = data_queue.get()
            if data is None:
                break
            st_time = time.time()
            is_rally_end, is_collide, main_pos3d, color, is_rally, frame_id = data
            if is_rally_end:
                c3f = visuals.Court3DFigure()
            try:
                if not is_rally and frame_id % 120 == 0:
                    c3f = visuals.Court3DFigure()
            except Exception as e:
                pass
            if main_pos3d is not None:
                c3f.add_pos3d(main_pos3d, color=color)
            # logger.debug(f"Process render_3d_court_process_funct rendering {main_pos3d} in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f})")
            rendered_queue.put(c3f.get_court_image())
        logger.info("Process render_3d_court_process_funct released")

    def render_3d_court_receive_funct(self, rendered_image_queue, frame_result: FrameResult):
        logger.debug(f"Frame {frame_result.frame_id} court 3d images received frame_result")
        rendered_image = rendered_image_queue.get().copy()
        frame_result.court_3d_image = rendered_image
        logger.debug(f"Frame {frame_result.frame_id} court 3d images got from rendered_image_queue")
        self.camera_render_done_syncers[frame_result] = Queue()
        for i in range(len(frame_result.cameras) - 1):
            self.camera_render_done_syncers[frame_result].put(False)
        self.camera_render_done_syncers[frame_result].put(True)
        for cam in self.cameras:
            logger.debug(f"Frame {frame_result.frame_id} court 3d images putting to camera {cam.name} render queue")
            self.camera_render_queues[cam].put(frame_result)
            logger.debug(f"Frame {frame_result.frame_id} court 3d images putted to camera {cam.name} render queue")
        logger.debug(f"Frame {frame_result.frame_id} court 3d images getting from rendered_image_queue")
        
        return frame_result

    def stop_camera_render_threads(self):
        for i in range(len(self.cameras) - 1):
            self.camera_render_finish_syncer.put(False)
        self.camera_render_finish_syncer.put(True)
        for cam in self.cameras:
            logger.debug(f"Stopping camera_render_thread_funct {cam.name}")
            self.camera_render_queues[cam].put(None)

    def camera_render_thread_funct(self, cam, cam_frame_render_queue, cam_frame_write_queue):
        while True:
            frame_result:FrameResult = cam_frame_render_queue.get()
            if frame_result is None:
                cam_frame_write_queue.put(None)
                if self.camera_render_finish_syncer.get():
                    self.on_result.send(None)
                break
            # self.render_info_cards(cam, frame_result)
            # self.render_3d_court(cam, frame_result)
            # self.render_plane_court(cam, frame_result)
            cam_frame_write_queue.put(frame_result)
            logger.debug(f"Frame {frame_result.frame_id} Camera {cam.name} render thread send to write thread (qlen={cam_frame_write_queue.qsize()})")
            if self.camera_render_done_syncers[frame_result].get():
                del self.camera_render_done_syncers[frame_result]  # !!important!! to prevent memory leak
                self.on_result.send(frame_result)
            
        logger.info(f"Camera {cam.name} render thread released")

    def render_infocards_stage_send_funct(self, cam, frame_queue, court_plane_image_queue, court_3d_queue, sponsors_image_queue, data_queue, frame_result: FrameResult):
        frame = frame_result.camera_result_frames[cam]
        frame_id = frame_result.frame_id
        if frame_result.event_is_rally:
            self.serve_speed_kmh = frame_result.serve_speed_kmh if frame_result.event_is_serve else self.serve_speed_kmh
            self.serve_cross_net_seconds = frame_result.serve_cross_net_seconds if frame_result.event_is_serve_cross_net else self.serve_cross_net_seconds
            self.spike_speed_kmh = frame_result.spike_speed_kmh if frame_result.event_is_spike else self.spike_speed_kmh
            self.spike_height_m = frame_result.spike_height_m if frame_result.event_is_spike else self.spike_height_m
        if frame_result.event_is_rally_reset:
            self.serve_speed_kmh = None
            self.serve_cross_net_seconds = None
            self.spike_speed_kmh = None
            self.spike_height_m = None
        vel_kmh, acc_ms2, jer_ms3 = frame_result.ball_vel_kmh, frame_result.ball_acc_ms2, frame_result.ball_jer_ms3
        
        court_plane_image = frame_result.court_plane_image
        court_3d_image = frame_result.court_3d_image
        is_have_court_plane_image = court_plane_image is not None
        is_have_court_3d_image = court_3d_image is not None
        is_have_sponser_image = self.sponsors_image is not None
        if True:
            is_have_court_3d_image = False
            is_have_court_plane_image = False

        data = [frame_id, cam.name, \
                self.serve_speed_kmh, self.serve_cross_net_seconds, self.spike_speed_kmh, self.spike_height_m, \
                frame_result.rally_tracker_state.name, \
                vel_kmh, acc_ms2, jer_ms3, \
                is_have_court_plane_image, is_have_court_3d_image, is_have_sponser_image]
        st_time = time.time()
        frame_queue.put(frame)
        if is_have_court_plane_image:
            court_plane_image_queue.put(court_plane_image)
        if is_have_court_3d_image:
            court_3d_queue.put(court_3d_image)
        if is_have_sponser_image:
            sponsors_image_queue.put(self.sponsors_image)
        data_queue.put(data)
        # logger.debug(f"Frame {frame_id} Camera {cam.name} infotext/card sent, put frame in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f})")
        return frame_result

    def render_infocards_stage_receive_funct(self, cam, rendered_frame_queue, frame_result: FrameResult):
        rendered_frame = rendered_frame_queue.get()
        frame_result.camera_result_frames[cam] = rendered_frame.copy()
        logger.debug(f"Frame {frame_result.frame_id} Camera {cam.name} infotext/card received")
        return frame_result

    @staticmethod
    def render_infocards_stage_process_funct(input_frame_queue, court_plane_image_queue, court_3d_image_queue, sponsors_image_queue, output_frame_queue, data):
        frame = input_frame_queue.get()
        # print(f"render_infocards_stage_process_funct rendering data {data}")
        frame_id, cam_name, serve_speed_kmh, serve_cross_net_seconds, spike_speed_kmh, spike_height_m, rally_state_str = data[:7]
        data = data[7:]
        vel_kmh, acc_ms2, jer_ms3 = data[:3]
        data = data[3:]
        is_have_court_plane_image = data[0]
        is_have_court_3d_image = data[1]
        is_have_sponsor_image = data[2]

        court_plane_image = None
        court_3d_image = None
        sponsors_image = None
        # if is_have_court_plane_image:
        #     court_plane_image = court_plane_image_queue.get()
        # if is_have_court_3d_image:
        #     court_3d_image = court_3d_image_queue.get()
        if is_have_sponsor_image:
            sponsors_image = sponsors_image_queue.get()

        st_time = time.time()
        text_y = 30
        # Disable for demo
        # cv.putText(frame, f"Frame {frame_id}", (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        text_y += 30
        # cv.putText(frame, f"Rally State: {rally_state_str}", (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        text_y += 30
        vel_kmh_str = f"{vel_kmh:.2f}" if vel_kmh is not None else "None"
        acc_ms2_str = f"{acc_ms2:.2f}" if acc_ms2 is not None else "None"
        jer_ms3_str = f"{jer_ms3:.2f}" if jer_ms3 is not None else "None"
        # cv.putText(frame, f"vel_kmh: {vel_kmh_str}", (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # text_y += 30
        # cv.putText(frame, f"acc_ms2: {acc_ms2_str}", (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # text_y += 30
        # cv.putText(frame, f"jer_ms3: {jer_ms3_str}", (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # text_y += 30
        serve_speed_kmh_str = f"{serve_speed_kmh:.0f}kmh" if serve_speed_kmh is not None else ""
        serve_cross_net_seconds_str = f"{serve_cross_net_seconds:.2f} s" if serve_cross_net_seconds is not None else ""
        spike_speed_kmh_str = f"{spike_speed_kmh:.0f}kmh" if spike_speed_kmh is not None else ""
        spike_height_m_str = f"{spike_height_m:.2f} m" if spike_height_m is not None else ""
        # cv.putText(frame, f"Serve Speed: {serve_speed_kmh_str}", (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # text_y += 30
        # cv.putText(frame, f"Spike Speed: {spike_speed_kmh_str}", (10, text_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
        # text_y += 30

        right_shift = 80
        is_eng = False
        if is_eng:  # Big change for unicode
            frame = visuals.draw_result_info_card(frame, (960 - 520 + right_shift, 1080 - 160 - 20), "Serve Speed", serve_speed_kmh_str)
            frame = visuals.draw_result_info_card(frame, (960 - 520 + right_shift, 1080 -  80 - 10), "Serve To Net", serve_cross_net_seconds_str)
            frame = visuals.draw_result_info_card(frame, (960 +   0 + 10 + right_shift, 1080 - 160 - 20), "Spike Speed", spike_speed_kmh_str)
            frame = visuals.draw_result_info_card(frame, (960 +   0 + 10 + right_shift, 1080 -  80 - 10), "Spike Height", spike_height_m_str)
        else:
            frame = visuals.draw_result_info_card(frame, (960 - 520 + right_shift, 1080 - 160 - 20), "發球速度", serve_speed_kmh_str, is_eng=False)
            frame = visuals.draw_result_info_card(frame, (960 - 520 + right_shift, 1080 -  80 - 10), "過網時間", serve_cross_net_seconds_str, is_eng=False)
            frame = visuals.draw_result_info_card(frame, (960 +   0 + 10 + right_shift, 1080 - 160 - 20), "殺球速度", spike_speed_kmh_str, is_eng=False)
            frame = visuals.draw_result_info_card(frame, (960 +   0 + 10 + right_shift, 1080 - 80 - 10), "殺球高度", spike_height_m_str, is_eng=False)
        if sponsors_image is not None:
            visuals.add_transparent_image(frame, sponsors_image, 
                                            0, frame.shape[0] - 150)
        # logger.info(f"Frame {frame_id} Camera {cam_name} infotext/card rendered in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f})")
        output_frame_queue.put(frame)
        return True

    def render_3d_court(self, cam, frame_result: FrameResult):
        frame_id = frame_result.frame_id
        frame = frame_result.camera_result_frames[cam]
        st_time = time.time()
        main_track2d = None
        VISUAL_PLOT_BOX = False
        if frame_result.main_track3d is not None:
            cam_track2ds = frame_result.main_track3d.cam_track2ds(frame_id)
            if (track2d:=cam_track2ds.get(cam)) is not None:
                main_track2d = track2d
                xywh = track2d.xywh(frame_id)
                color = [int(c*255) for c in self.cmap_track_id(frame_result.main_track3d.track_id)]
                if xywh is not None:
                    xyxy = np.hstack((xywh[:2] - xywh[2:] / 2, xywh[:2] + xywh[2:] / 2))
                    # visuals.plot_one_box(xyxy ,frame, color, label=f"Main {frame_result.main_track3d.track_id}")
                    ball_vel_kmh_str = f"{int(frame_result.ball_vel_kmh)}" if frame_result.ball_vel_kmh is not None else ""
                    for i in range(0):
                        if (_xywh := track2d.xywh(frame_id - i - 1)) is not None:
                            _xyxy = np.hstack((_xywh[:2] - _xywh[2:] / 2, _xywh[:2] + _xywh[2:] / 2))
                            _color = [int(c*0.5) for c in color]
                            if VISUAL_PLOT_BOX:
                                visuals.plot_one_box(_xyxy ,frame, _color, label=f"{ball_vel_kmh_str} kmh")
                    if VISUAL_PLOT_BOX:
                        visuals.plot_one_box(xyxy ,frame, color, label=f"{ball_vel_kmh_str} kmh")

        # Draw the non-main track2d
        for alive_track in frame_result.camera_active_track2ds[cam].values():
            if alive_track == main_track2d:
                continue
            xywh = alive_track.xywh(frame_id)
            # logger.info(f"Track {alive_track.track_id} xywh: {xywh}")
            color = [int(c*255) for c in self.cmap_track_id(alive_track.track_id)]
            # make the color low saturation
            color = [int(c * 0.5 + 0.5 * 255) for c in color]
            if xywh is not None:
                xyxy = np.hstack((xywh[:2] - xywh[2:] / 2, xywh[:2] + xywh[2:] / 2))
                # DEBUG_VISUAL Disable for demo
                # visuals.plot_one_box(xyxy ,frame, color, label=f"Track {alive_track.track_id}")
        # Draw the filtered track2d
        for filtered_track in frame_result.camera_filtered_track2ds[cam].values():
            xywh = filtered_track.xywh(frame_id)
            color = [int(c*255) for c in self.cmap_track_id(filtered_track.track_id)]
            # make the color low saturation
            color = [int(c * 0.5 + 0.5 * 255) for c in color]
            if xywh is not None:
                xyxy = np.hstack((xywh[:2] - xywh[2:] / 2, xywh[:2] + xywh[2:] / 2))
                # DEBUG_VISUAL Disable for demo
                # visuals.plot_one_box(xyxy ,frame, color, label=f"Filt {filtered_track.track_id}")

        lines = frame_result.camera_epilines.get(cam)
        if lines is not None:
            for track2d, line in lines.items():
                color = [int(c*255) for c in self.cmap_track_id(track2d.track_id)]
                if track2d != main_track2d:
                    color = [int(c * 0.5 + 0.5 * 255) for c in color]
                # DEBUG_VISUAL Disable for demo
                # visuals.draw_line(frame, line, color)
        # logger.debug(f"Frame {frame_id} Camera {cam.name} track2d rendered in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f})")
        st_time = time.time()
        # Note: Do the simple paste here,
        # the reason no doing by multi-processing is because the cost to copy the image to 
        # shared memory is just the same as pasting it directly (write to background's memory).
        court_plane_image = frame_result.court_plane_image
        court_3d_image = frame_result.court_3d_image
        # Disable for WMG
        # if court_plane_image is not None:
        #     frame[0:court_plane_image.shape[0], frame.shape[1]-court_plane_image.shape[1]:frame.shape[1]] = court_plane_image
        if court_3d_image is not None:
            cv.rectangle(frame, (frame.shape[1] - court_3d_image.shape[1], frame.shape[0] - court_3d_image.shape[0]),
                            (frame.shape[1], frame.shape[0]), (250, 250, 250), -1)
            visuals.paste_image(frame, court_3d_image,
                                frame.shape[1] - court_3d_image.shape[1],
                                frame.shape[0] - court_3d_image.shape[0])
            # logger.info(f"Frame {frame_id} Camera {cam.name} c3d pasted in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f})")
        # logger.debug(f"Frame {frame_id} Camera {cam.name} 2d3d pasted in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f})")
        return frame_result
    
    def render_plane_court(self, cam, frame_result: FrameResult):
        st_time = time.time()
        frame_id = frame_result.frame_id
        frame = frame_result.camera_result_frames[cam]
        # court_plane_image = frame_result.court_plane_image
        # Disable WMG
        # frame[0:court_plane_image.shape[0], frame.shape[1]-court_plane_image.shape[1]:frame.shape[1]] = court_plane_image
        if self.sponsors_image is not None:
            visuals.add_transparent_image(frame, self.sponsors_image, 
                                            0, frame.shape[0] - 150)
        
        if frame_result.frame_id % 10 == 0 and self.is_video_mode and False:
            if frame_result.court_3d_image is not None:
                cv.imwrite(f"./temp/c3f/c3f-{frame_result.frame_id}.png", frame_result.court_3d_image)
            cv.imwrite(f"./temp/cam/cam-{cam.name}-{frame_result.frame_id}.png", frame)
            pass
        # logger.info(f"Frame {frame_id} Camera {cam.name} c2d pasted in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f})")
        return frame_result
    
    def camera_write_thread_funct(self, cam, queue):
        while True:
            frame_result: FrameResult = queue.get()
            st_time = time.time()
            if frame_result is None:
                if (old_cam_out_path:=self.camera_output_paths.get(cam)) is not None \
                    and (cam_out:=self.output_path_outs.get(old_cam_out_path)) is not None:
                    # cam_out.release() # cv2
                    logger.info(f"Camera {cam.name} output path {old_cam_out_path} out released")
                    cam_out.close()
                    del self.output_path_outs[old_cam_out_path]
                logger.info(f"Camera {cam.name} write thread received None, stop writing")
                break
            logger.info(f"Frame {frame_result.frame_id} Camera {cam.name} camera write thread received")
            frame = frame_result.camera_result_frames[cam]
            if not self.is_video_mode and not frame_result.event_is_rally:
                # enable prepend buffer method when stream mode
                logger.debug(f"Frame {frame_result.frame_id} Camera {cam.name} prepending frame")
                self.camera_prepend_buffers[cam].append(frame)
                logger.debug(f"Frame {frame_result.frame_id} Camera {cam.name} prepending frame done")
                if len(self.camera_prepend_buffers[cam]) > self.prepend_length:
                    logger.debug(f"Frame {frame_result.frame_id} Camera {cam.name} prepending frame poping")
                    self.camera_prepend_buffers[cam].pop(0)
                    logger.debug(f"Frame {frame_result.frame_id} Camera {cam.name} prepending frame poping done")

                # tvl20 hotfix
                old_cam_out_path = self.camera_output_paths.get(cam)
                if old_cam_out_path is not None:
                    st_time_out = time.time()
                    self.output_path_outs[old_cam_out_path].close()
                    del self.output_path_outs[old_cam_out_path]
                    del self.camera_output_paths[cam]
                    logger.info(f"Camera {cam.name} output path {old_cam_out_path} out released using free time in {time.time() - st_time_out:.2f}")
                continue

            logger.debug(f"Frame {frame_result.frame_id} Camera {cam.name} is a rally, writing frame")
            frame_id = frame_result.frame_id
            cam_out_path = frame_result.camera_output_paths[cam]
            old_cam_out_path = self.camera_output_paths.get(cam)
            if old_cam_out_path is None or old_cam_out_path != cam_out_path:
                if old_cam_out_path is not None:
                    st_time_out = time.time()
                    self.output_path_outs[old_cam_out_path].close()
                    del self.output_path_outs[old_cam_out_path]
                    logger.info(f"Camera {cam.name} output path {old_cam_out_path} out released in {time.time() - st_time_out:.2f}")

                logger.info(f"Creating Writer for Camera {cam.name} output path: {cam_out_path}") # ultrafast
                
                out = WriteGear(output=str(cam_out_path), compression_mode=True, logging=False, \
                                **{"-vcodec":"libx264", "-crf":18, "-preset":"ultrafast", "-pix_fmt":"yuv420p", \
                                "-input_framerate": self.fps, "-output_dimensions": self.video_size})
                logger.info(f"Camera {cam.name} output path changed to {cam_out_path}, new out created")
                
                
                self.camera_output_paths[cam] = cam_out_path
                self.output_path_outs[cam_out_path] = out
            else:
                out = self.output_path_outs[cam_out_path]
            # !!!Temporarily disable prepend buffer method!!! Refactoring it to RallyTracker
            # for prep_idx, prepend_frame in enumerate(self.camera_prepend_buffers[cam]):
            #     logger.debug(f"Frame {frame_id} Camera {cam.name} writing prepend frame {prep_idx}")
            #     out.write(prepend_frame)
            out.write(frame)
            self.camera_prepend_buffers[cam] = []
            # logger.info(f"Frame {frame_id} Camera {cam.name} written in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f})")

class RealtimeDisplay:
    """
        For displaying realtime result,
        the GUI library tent to be running in main thread, so this class is not neccessary to be threaded.
        Currently this is a callback handler, let the main thread bind a callback to get frames.
    """
    on_display_frame = Signal(tuple)  # Camera, frame_id, frame
    def __init__(self, video_size, fps, display_cameras) -> None:
        self.video_size = video_size
        self.start_time = None
        self.fps = fps
        self.display_cameras = display_cameras if type(display_cameras) is list else [display_cameras]
        self.frame_result_queue = Queue(maxsize=10)
        self.dispatch_thread = Thread(target=self.dispatch_thread_funct, args=(self.frame_result_queue,))
        self.dispatch_thread.start()

    def on_frame(self, frame_result: FrameResult):
        self.frame_result_queue.put(frame_result)
        if frame_result is None:
            return
        if self.start_time is None:
            self.start_time = time.time()
        elif frame_result.frame_id % 100 == 0:
            elapsed_time = time.time() - self.start_time
            logger.info(f"Full Video Receive: Qlen({self.frame_result_queue.qsize()}) elapsed time: {elapsed_time:.2f}, FPS: {frame_result.frame_id / elapsed_time:.2f}")

    def dispatch_thread_funct(self, frame_result_queue):
        while True:
            frame_result: FrameResult = frame_result_queue.get()
            if frame_result is None:
                self.on_display_frame.send(None)
                break
            for cam, frame in frame_result.camera_result_frames.items():
                if cam in self.display_cameras:
                    self.on_display_frame.send((cam, frame_result.frame_id, frame))

class FullVideoWriter:
    """
        For write full game video
    """
    def __init__(self, camera_video_paths, video_size, fps) -> None:
        self.camera_video_paths = camera_video_paths
        self.video_size = video_size
        self.start_time = None
        self.fps = fps
        self.camera_outs = {}
        self.camera_write_threads = {}
        self.camera_write_queues = {}
        self.frame_result_queue = Queue(maxsize=10)
        self.dispatch_thread = Thread(target=self.dispatch_thread_funct, args=(self.frame_result_queue,))
        self.dispatch_thread.start()
        for cam, path in camera_video_paths.items():
            if path is not None:
                # out = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*'mp4v'), fps, video_size)
                out = WriteGear(output=str(path), compression_mode=True, logging=False, \
                                **{"-vcodec":"libopenh264", "-crf":18, "-preset":"ultrafast", "-pix_fmt":"yuv420p", \
                                   "-input_framerate": self.fps, "-output_dimensions": self.video_size})
                self.camera_outs[cam] = out
                self.camera_write_queues[cam] = q = Queue(maxsize=10)
                self.camera_write_threads[cam] = Thread(target=self.camera_write_thread_funct, args=(cam, q, out))
            else:
                self.camera_outs[cam] = None
        
        self.last_show_time = None
        self.is_window_created = False
        for t in self.camera_write_threads.values():
            t.start()

    def on_frame(self, frame_result: FrameResult):
        self.frame_result_queue.put(frame_result)
        if frame_result is None:
            return
        if self.start_time is None:
            self.start_time = time.time()
        elif frame_result.frame_id % 100 == 0:
            elapsed_time = time.time() - self.start_time
            logger.info(f"Full Video Receive: Qlen({self.frame_result_queue.qsize()}) elapsed time: {elapsed_time:.2f}, long-FPS: {frame_result.frame_id / elapsed_time:.2f}")

    def dispatch_thread_funct(self, frame_result_queue):
        while True:
            frame_result: FrameResult = frame_result_queue.get()
            if frame_result is None:
                for cam, q in self.camera_write_queues.items():
                    q.put(None)
                break
            for cam, q in self.camera_write_queues.items():
                q.put(frame_result)

    def camera_write_thread_funct(self, cam, queue, out):
        while True:
            frame_result: FrameResult = queue.get()
            if frame_result is None:
                break
            st_time = time.time()
            frame = frame_result.camera_result_frames.get(cam)
            if frame is not None and out is not None:
                out.write(frame)
                # logger.info(f"Frame {frame_result.frame_id} Camera {cam.name} Full Video written in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f}), Qlen({queue.qsize()})")
        if out is not None:
            # out.release()
            out.close()

class ResultSerializer:
    def __init__(self) -> None:
        logger.info("ResultSerializer initialized")
        self.rally_dir = None
        self.init_data()
        self.frame_queue = Queue()
        self.add_thread = Thread(target=self.add_frame_result_thread_funct, args=(self.frame_queue,))
        self.json_queue = multiprocessing.Queue()
        self.write_process = multiprocessing.Process(target=self.write_data_process_funct, args=(self.json_queue,))
        self.add_thread.start()
        self.write_process.start()

    def init_data(self):
        self.start_frame_id = None
        self.ball_data_list = []
        self.event_data_list = []

    def write_data(self):
        if self.rally_dir is None:
            logger.warning("No rally dir to write")
            return
        rally_dir = self.rally_dir
        rally_dir.mkdir(exist_ok=True)
        data = {
            "ball_data": self.ball_data_list,
            "event": self.event_data_list,
        }
        json_path = rally_dir / "ball_data.json"
        self.json_queue.put((json_path, data))
    
    @staticmethod
    def write_data_process_funct(json_queue):
        while True:
            data = json_queue.get()
            if data is None:
                break
            json_path, data = data
            st_time = time.time()
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
            # logger.info(f"Rally serialized to {json_path}, in {time.time() - st_time:.4f} s (FPS {1 / (time.time() - st_time):.2f})")
        logger.info("ResultSerializer finished")

    def add_frame_result(self, frame_result: FrameResult):
        rally_frame_id = frame_result.frame_id - self.start_frame_id + 1
        if frame_result.main_pos3d is not None \
            and frame_result.main_track3d is not None:
            main_track3d: Track3D = frame_result.main_track3d
            camera_bboxes = {}
            for cam, track2d in main_track3d.cam_track2ds(frame_result.frame_id).items():
                camera_bboxes[cam.name] = try_tolist(track2d.xywh(frame_result.frame_id))
            ball_data = {
                "frame_id": rally_frame_id,
                "pos3d": try_tolist(frame_result.main_pos3d),
                "color": try_tolist(frame_result.main_color),
                "vel_kmh": try_tolist(frame_result.ball_vel_kmh),
                "acc_ms2": try_tolist(frame_result.ball_acc_ms2),
                "jer_ms3": try_tolist(frame_result.ball_jer_ms3),
                ### ----------------------------------------------- New Add ------------------------------------------------------- ###
                "spin_rpm": try_tolist(frame_result.spin_rate_rpm),
                "state": try_tolist(frame_result.rally_tracker_state.name),
                ### --------------------------------------------------------------------------------------------------------------- ###
                "camera_bboxes": camera_bboxes,
            }
            self.ball_data_list.append(ball_data)

        if frame_result.event_is_collide:
            self.event_data_list.append({"frame_id": rally_frame_id, "event": "collide"})
        if frame_result.event_is_serve:
            self.event_data_list.append({"frame_id": rally_frame_id, "event": "serve", 
                                         "serve_speed_kmh": frame_result.serve_speed_kmh})
        if frame_result.event_is_serve_cross_net:
            self.event_data_list.append({"frame_id": rally_frame_id, "event": "serve_cross_net",
                                         "serve_cross_net_seconds": frame_result.serve_cross_net_seconds})
        if frame_result.event_is_spike:
            self.event_data_list.append({"frame_id": rally_frame_id, "event": "spike",
                                         "spike_speed_kmh": frame_result.spike_speed_kmh,
                                         "spike_height_m": frame_result.spike_height_m})
        if frame_result.event_is_attack:
            self.event_data_list.append({"frame_id": rally_frame_id, "event": "attack", 
                                         "attack_speed_kmh": frame_result.attack_speed_kmh,
                                         "attack_height_m": frame_result.attack_height_m})
            
    def add_frame_result_thread_funct(self, frame_queue):
        while True:
            frame_result = frame_queue.get()
            if frame_result is None:
                self.write_data()
                self.json_queue.put(None)
                break
            logger.info(f"Frame {frame_result.frame_id} rally serializer received")
            if frame_result.event_is_rally_begin:
                self.init_data()
                self.start_frame_id = frame_result.frame_id
                self.rally_dir = frame_result.rally_dir
            if frame_result.event_is_rally:
                self.add_frame_result(frame_result)
            if frame_result.event_is_rally_end:
                self.write_data()

    def on_frame(self, frame_result: FrameResult):
        self.frame_queue.put(frame_result)

class DetectSystem:
    # This class is the main class that holds all the modules
    def __init__(self, model_path, model_imgsz, cameraset, camera_name_inputs, start_time, output_dir, fps=60, video_size=(1920, 1080), 
                 is_video_mode=True, is_device_mode=False, is_write_full=False, is_display_mode=False, simulate_fps=None, is_sponsor_mode=True,
                 det_log_level=None, is_debug=True) -> None:
        self.fps = fps
        self.video_size = video_size
        self.model_imgsz = model_imgsz
        self.is_video_mode = is_video_mode
        self.is_device_mode = is_device_mode
        self.is_display_mode = is_display_mode
        self.is_sponsor_mode = is_sponsor_mode
        self.is_debug = is_debug
        self.model_path = model_path
        self.cameraset = cameraset
        self.cameras = cameraset.cameras
        self.camera_name_inputs = camera_name_inputs
        self.camera_source_infos = {}
        self.camera_name_camera_map = {cam.name: cam for cam in self.cameras}
        self.camera_inputs = {self.camera_name_camera_map[cam_name]: input_path for cam_name, input_path in camera_name_inputs.items()}
        is_camera_reader_process = True
        print(f"FLAG: is_camera_reader_process: {is_camera_reader_process}")
        for camera_name, input_path in camera_name_inputs.items():
            camera = self.camera_name_camera_map.get(camera_name)
            if is_device_mode:
                source_info = SourceInfo(str(input_path), fps, video_size[0], video_size[1], source_type=SourceType.DEVICE)
            else:
                source_info = SourceInfo(str(input_path), fps, video_size[0], video_size[1], source_type=SourceType.FILE)
            self.camera_source_infos[camera] = source_info

        if not is_camera_reader_process:  # original threaded CameraReader
            self.camera_reader = CameraReader(self.cameras, self.camera_source_infos, start_time, self.fps, simulate_fps=simulate_fps)
        else:
            self.camera_reader = CameraReader(self.cameras, self.camera_source_infos, start_time, self.fps, simulate_fps=simulate_fps, \
                is_process=True)
        self.ball_detector = BallDetector(model_path, model_imgsz, det_log_level=det_log_level)
        self.ball_tracker = BallTracker(self.cameraset, self.cameras, self.fps, self.model_imgsz, self.video_size)
        self.rally_tracker = RallyTracker(self.fps, is_video_mode=is_video_mode)
        self.serializer = ResultSerializer()
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        self.rally_output_manager = RallyOutputManager(output_dir, self.camera_inputs, is_video_mode=is_video_mode)
        self.visualizer = ResultRenderer(self.cameras, fps=fps, is_video_mode=is_video_mode, is_sponsor_mode=is_sponsor_mode)
        # self.visualizer = ResultRenderer([self.cameras[0]], fps=fps, is_video_mode=is_video_mode)
        # self.visualizer = ResultRenderer([], fps=fps, is_video_mode=is_video_mode)
        
        self.camera_reader.on_result.connect(self.ball_detector.on_frame)
        self.ball_detector.on_result.connect(self.ball_tracker.on_frame)
        self.ball_tracker.on_result.connect(self.rally_tracker.on_frame)
        self.rally_tracker.on_result.connect(self.rally_output_manager.on_frame)
        self.rally_output_manager.on_result.connect(self.visualizer.on_frame)
        self.visualizer.on_result.connect(self.serializer.on_frame)
        
        if is_write_full:
            self.camera_output_paths = {}
            for cam, input_path in self.camera_inputs.items():
                if is_device_mode:
                    full_output_path = output_dir / f"Full_device_{str(input_path)}.mp4"
                else:
                    full_output_path = output_dir / f"Full_{Path(input_path).name}.mp4"
                self.camera_output_paths[cam] = full_output_path
            self.full_video_writer = FullVideoWriter(self.camera_output_paths, video_size, fps)
            self.visualizer.on_result.connect(self.full_video_writer.on_frame)
        else:
            self.full_video_writer = None
        
        if is_display_mode:
            self.realtime_display = RealtimeDisplay(video_size, fps, self.cameras[0])
            self.visualizer.on_result.connect(self.realtime_display.on_frame)
        else:
            self.realtime_display = None

    # THE ENTRY POINT
    def run(self):
        self.camera_reader.run()

def get_hdr80_video_paths(video_path, camera_names):
    VIDEOFILE_PATTERN = r"HDR80_(?P<camera>[^_]*)_Live_(?P<year>.{4})(?P<date>.{4})_(?P<time>.{6})_000.mov"
    VIDEOFILE_REGEX = re.compile(VIDEOFILE_PATTERN)
    match = VIDEOFILE_REGEX.match(Path(video_path).name)
    video_dir = Path(video_path).parent
    video_paths = {}
    for camera_name in camera_names:
        st_idx, ed_idx = match.span('camera')
        filename = str(match.string)[:st_idx] + camera_name + str(match.string)[ed_idx:]
        video_paths[camera_name] = video_dir / filename
    return video_paths

def callback_display_frame(queue, data):
    if data is None:
        queue.put(None)
        return
    camera, frame_id, frame = data
    logger.debug(f"Display frame callback received Frame {frame_id}")
    queue.put(frame)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loglevel", type=str, default="DEBUG", help="Log level")
    parser.add_argument("--detloglevel", type=str, default="WARNING", help="Detector Process Log level")
    parser.add_argument("--source", type=str, default="./data/HDR80_A_Live_20230211_153630_000.mov", help="Video source path or device index")
    parser.add_argument("--source_size", type=str, default="1920,1080", help="Video source size")
    parser.add_argument("--is_device", action="store_true", help="Is the 'source' device index")
    parser.add_argument("--cameras", type=str, default="A,D", help="Camera names")
    parser.add_argument("--camset", type=str, default="./camsets/camset_0211", help="CameraSet directory")
    parser.add_argument("--model", type=str, default="yolov8n_mikasa_1280_v1.pt", help="Model path")
    parser.add_argument("--imgsz", type=str, default="1280", help="Model input size, can be [w,h] or int")
    parser.add_argument("--stream", action="store_true", help="Stream mode")
    parser.add_argument("--outdir", type=str, default="./result", help="Output directory, when stream mode, it's the rally output directory")
    parser.add_argument("--display", action="store_true", help="Show a window for demo")
    parser.add_argument("--simfps", default=None, help="Simulate stream FPS for video mode (for debug)")
    parser.add_argument("--write-full", action="store_true", help="Write full game video, format is 'Full_{source path/device name}.mp4'")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--sponsor", action="store_true", help="Add sponsor logo in the video")
    parser.add_argument("--fs", action="store_true", help="Display in fullscreen mode")
    args = parser.parse_args()
    debug_mode = args.debug
    sponsor_mode = args.sponsor
    fullscreen_mode = args.fs
    video_size = [int(s) for s in args.source_size.split(",")]
    model_imgsz = [int(args.imgsz), int(args.imgsz)] if ',' not in args.imgsz else [int(s) for s in args.imgsz.split(",")]
    # model_imgsz = [model_imgsz[0], int(model_imgsz[0] * video_size[1] / video_size[0])]
    logger.debug(f"Adjusted model_imgsz: {model_imgsz}")
    # logger.add(sys.stdout, level="SUCCESS")
    log_level = args.loglevel
    det_log_level = args.detloglevel
    is_video_mode = not args.stream
    camera_video_path = args.source
    cameraset_dir = args.camset
    camera_names = args.cameras.split(",")
    model_path = args.model
    output_dir = args.outdir
    is_device_mode = args.is_device
    is_display_mode = args.display
    is_write_full = args.write_full
    simulate_fps = int(args.simfps) if args.simfps is not None else None
    assert (not (is_video_mode and is_device_mode)), "Video mode and device mode can not be both true"
    logger.remove()  # remove the old handler. Else, the old one will work along with the new one you've added below'
    logger.add(sys.stdout, level=log_level)

    cameraset_loader = CameraSetFileCalibrator(cameraset_dir)
    cameraset = cameraset_loader.load().get_camset()
    cameras = [cameraset.get_camera(cam) for cam in camera_names]
    cameraset.cameras = cameras

    logger.info(f"CameraSet Cameras: {cameraset.cameras}")
    cameraset.summary()
    if is_device_mode:
        start_time = datetime.datetime.now()
        device_indexes = [int(d) for d in camera_video_path.split(',')]
        camera_inputs = {cam_name:device for cam_name, device in zip(camera_names, device_indexes)}
    else:
        regex_match = VIDEOFILE_REGEX.match(Path(camera_video_path).stem)
        start_time = hdr80_match_to_datetime(regex_match)
        camera_inputs = get_hdr80_video_paths(camera_video_path, camera_names)
    ds = DetectSystem(model_path, model_imgsz, cameraset, camera_inputs, start_time, output_dir, video_size=video_size, is_video_mode=is_video_mode, is_device_mode=is_device_mode,
                      is_write_full=is_write_full, is_display_mode=is_display_mode, simulate_fps=simulate_fps, det_log_level=det_log_level)
    st_time = time.time()
    ds_thread = Thread(target=ds.run)
    display_frame_queue = Queue(maxsize=300)
    if ds.realtime_display is not None:
        cb = partial(callback_display_frame, display_frame_queue)
        ds.realtime_display.on_display_frame.connect(cb)
    
    ds_thread.start()
    skip_frame_counter = 0  # for slack == 1.0
    if ds.realtime_display is not None:
        last_display_time = None
        WIN_NAME = "Display"
        DISPLAY_WIDTH = 1600
        cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
        cv.resizeWindow(WIN_NAME, DISPLAY_WIDTH, int(DISPLAY_WIDTH * video_size[1] / video_size[0]))
        if fullscreen_mode:
            cv.setWindowProperty(WIN_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        while True:
            frame = display_frame_queue.get()
            if frame is None:
                break
            skip_frame_counter += 1
            if skip_frame_counter > 6:
                skip_frame_counter = 0
                continue
            # frame = imutils.resize(frame, width=1600, inter=cv.INTER_LINEAR)
            logger.debug(f"Display frame: {frame.shape}")
            cv.imshow(WIN_NAME, frame)
            if last_display_time is None:
                last_display_time = time.time()
                cv.waitKey(1)
            else:
                # wait_time_scale_size = 1.75
                wait_time_scale_size = 1.5
                slack = display_frame_queue.qsize() / display_frame_queue.maxsize
                wait_time_scale = 1 + wait_time_scale_size * (0.5 - slack)
                ms_per_frame = int(1000 / ds.fps)
                wait_time = ms_per_frame - int((time.time() - last_display_time) * 1000)
                if wait_time < -ms_per_frame:
                    for i in range(int(-wait_time / ms_per_frame)):
                        logger.warning(f"Display skipping frame {i}")
                        if display_frame_queue.get() is None:
                            break
                else:
                    wait_time = max(0, int(wait_time * wait_time_scale))
                    logger.info(f"Display Wait time: {wait_time} ms, dqlen: {display_frame_queue.qsize()}, slack: {slack:.2f}, wait_time_scale: {wait_time_scale:.2f}")
                    if wait_time > 0:
                        wait_time = max(1, wait_time)
                        cv.waitKey(wait_time)
                last_display_time = time.time()
    cv.destroyAllWindows()

    logger.info("Main thread waiting for ds_thread to join")
    ds_thread.join()
    logger.info("Main thread joined ds_thread")
    logger.info(f"Main thread waiting for display_frame_queue to empty")
    ds.visualizer.join()
    logger.info(f"Main thread joined display_frame_queue")
    logger.success(f"Total time: {time.time() - st_time:.2f} s")

if __name__ == "__main__":
    main()