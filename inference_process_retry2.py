from ultralytics import YOLO
import multiprocessing
import threading
import time
import cv2 as cv
import numpy as np
from vidgear.gears import WriteGear, VideoGear
from arrayqueues import ArrayQueue
import argparse
import torch
import queue
from pathlib import Path
from functools import partial
import os

class YOLOPredictor:
    def __init__(self, model_path, device='cpu', predict_params=None) -> None:
        self.device = device
        if type(self.device) in [str, int] and self.device != 'cpu':
            self.device = torch.device(f"cuda:{self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.predict_params = predict_params if predict_params is not None else {}

    def predict(self, frame):
        results = self.model.predict(frame, device=self.device, **self.predict_params, verbose=False)
        return results
    
    def __call__(self, frame):
        return self.predict(frame)
    
class Box:
    xywh = None
    conf = None
    cls = None

class FrameResult:
    camera_frames = {}
    camera_boxes = {}

class ProcessPoolPredictor:
    def __init__(self, create_model, num_process_per_device=1, devices=['0']):
        # create_model(device) should return a model
        self.num_process_per_device = num_process_per_device
        self.devices = devices
        self.processes = []
        self.input_queues = []
        self.input_key_queues = []
        self.output_queue_lock = multiprocessing.Lock()
        self.output_queue = multiprocessing.Queue()
        self.output_key_queue = multiprocessing.Queue()
        self.stopped_processes = 0

        for device in devices:
            for i in range(num_process_per_device):
                input_np_queue = ArrayQueue(max_mbytes=4096)
                input_key_queue = multiprocessing.Queue()
                self.input_queues.append(input_np_queue)
                self.input_key_queues.append(input_key_queue)
                process = multiprocessing.Process(target=self._consumer, args=(create_model, device, input_np_queue, input_key_queue, self.output_queue, self.output_key_queue))
                self.processes.append(process)
                
        for process in self.processes:
            process.start()
        print(f"Started {len(self.processes)} processes")
        self.key = 0
        self.get_key = 0
        self.results = {}
    
    def _consumer(self, create_model, device, input_queue, input_key_queue, output_queue, output_key_queue):
        model = create_model(device)
        print(f"Process on device {device}, PID: {os.getpid()} model created")
        while True:
            key = input_key_queue.get()
            if key is None:
                output_queue.put((None, None))
                break
            frame = input_queue.get()
            print(f"Predict {key} in {os.getpid()} on device {device}")
            result = model(frame)
            boxes = result[0].boxes.cpu().numpy()
            box_list = []
            for box in boxes:
                b = Box()
                b.xywh = box.xywh
                b.conf = box.conf
                b.cls = box.cls
                box_list.append(b)
            output_queue.put((key, boxes))
            # output_key_queue.put(key)
        print(f"Process {os.getpid()} on device {device} stopped")
    
    def predict(self, frame):
        key = self.key
        self.key += 1
        # Find min queue
        min_queue = self.input_queues[0]
        min_key_queue = self.input_key_queues[0]
        min_len = min_queue.qsize()
        min_idx = 0
        for i in range(1, len(self.input_queues)):
            if self.input_queues[i].qsize() < min_len:
                min_len = self.input_queues[i].qsize()
                min_queue = self.input_queues[i]
                min_key_queue = self.input_key_queues[i]
                min_idx = i
        while True:
            try:
                min_queue.put(frame)
                break
            except Exception as e:
                pass
        min_key_queue.put(key)
        return key

    def stop(self):
        for i in range(len(self.processes)):
            self.input_key_queues[i].put(None)
            # self.input_queues[i].put(None)
    
    def join(self):
        print('Joining processes')
        for process in self.processes:
            if process.is_alive():
                process.join()
        print('All processes joined')
    
    def get_result(self):
        if self.stopped_processes == len(self.processes):
            return None
        key = self.get_key
        self.get_key += 1
        while key not in self.results:
            # res_key = self.output_key_queue.get()
            # res = self.output_queue.get()
            res_key, res = self.output_queue.get()
            print(f"Get result {res_key}")
            if res_key is None:
                self.stopped_processes += 1
                if self.stopped_processes == len(self.processes):
                    return None
                continue
            self.results[res_key] = res
            
        return self.results[key]
                

def main():
    video_path = 'data/HDR80_A_Live_20230211_153630_000.mov'
    video_reader = VideoGear(source=video_path, logging=True).start()
    output_path = Path('temp/output.mp4')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_writer = WriteGear(output=output_path, logging=True)
    model_path = 'yolov8n_mikasa_1280_v1.pt'
    num_process_per_device = 4
    # devices = ['0', '1']
    devices = ['1']

    def create_consumer(device='0'):
        predictor = YOLOPredictor(model_path, device=device)
        def consumer(data):
            frame = data
            results = predictor(frame)
            return results
        return consumer
    def postprocess(result):
        boxes = result[0].boxes.cpu().numpy()
        box_list = []
        for box in boxes:
            b = Box()
            b.xywh = box.xywh
            b.conf = box.conf
            b.cls = box.cls
            box_list.append(b)
        return box_list
    consumers = []

    print('Start consuming')
    st_time = time.time()
    process_pool_predictor = ProcessPoolPredictor(partial(create_consumer), num_process_per_device, devices)

    def read_funct(cb):
        frame_count = 0
        while True:
            st_time = time.time()
            frame = video_reader.read()
            if frame is None:
                break
            frame_count += 1
            # print(f'Read frame {frame_count}, FPS: {frame_count/(time.time()-st_time):.2f}')
            # print(f'Read frame {frame_count}, FPS: {frame_count/(time.time()-st_time):.2f}')
            key = process_pool_predictor.predict(frame)

        if callable(cb):
            cb()
        print('read_funct end')
    read_thread = threading.Thread(target=read_funct, args=(process_pool_predictor.stop,))
    read_thread.start()
    frame_count = 0
    st_time = time.time()
    while True:
        # frame = frame_queue.get()
        result = process_pool_predictor.get_result()
        if result is None:
            break
        # video_writer.write()
        # print(results)
        frame_count += 1
        elapsed_time = time.time() - st_time
        print(f'Elapsed time: {elapsed_time:.2f}, FPS: {frame_count/elapsed_time:.2f}, {len(result)} boxes')
    video_reader.stop()
    if read_thread.is_alive():
        print(f"Joining read thread")
        read_thread.join()

    print('Read thread stopped')
    process_pool_predictor.join()
    print('All processes stopped')
    

if __name__ == '__main__':
    main()
