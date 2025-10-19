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


class QueueConsumer:
    """
        Process manager, use hybrid of multiprocessing and threading to manage the process,
        use thread to pass data_id to make user use any type as data_id (it needs to be hashable)
    """
    def __init__(self, create_consumer, postprocess, input_queue=None, output_queue=None, max_mbytes=1024) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        if self.input_queue is None:
            self.input_queue = ArrayQueue(max_mbytes=max_mbytes)
        if self.output_queue is None:
            self.output_queue = multiprocessing.Queue()
        self.process_queue = ArrayQueue(max_mbytes=max_mbytes)
        self.process = None
        self.create_consumer = create_consumer
        self.postprocess = postprocess
        if not callable(self.create_consumer):
            raise ValueError("create_consumer must be a callable function")
        if not callable(postprocess):
            raise ValueError("postprocess must be a callable function")
        
    def start(self):
        if self.process is not None and self.process.is_alive():
            return self
        self.process = multiprocessing.Process(target=self.consume, args=(self.input_queue, self.output_queue, self.create_consumer, self.postprocess))
        self.process.start()
        return self

    def stop(self):
        if self.process is None or not self.process.is_alive():
            return
        self.input_queue.put(None)
        self.process.join()
        self.process.close()
    
    @staticmethod
    def consume(input_queue, output_queue, create_consumer, postprocess):
        consumer = create_consumer()
        while True:
            data = input_queue.get()
            # print(f"Consumer {self.process.pid} got data")
            if data is None:  # sentinel value to break
                break
            st_time = time.time()
            data = consumer(data)
            data = postprocess(data)
            output_queue.put(data)
            # print(f"Consumer {multiprocessing.current_process().pid} put data, FPS: {1/(time.time()-st_time):.2f}")

class QueueConsumerPool:
    def __init__(self, consumers, input_queue=None, output_queue=None, max_mbytes=1024) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        if self.input_queue is None:
            self.input_queue = multiprocessing.Queue()
        if self.output_queue is None:
            self.output_queue = multiprocessing.Queue()
        self.next_key = 0
        self.key_queue = multiprocessing.Queue()
        self.consumers = consumers

    def start(self):
        for consumer in self.consumers:
            consumer.start()
        self.input_thread = threading.Thread(target=self.consume_input).start()
        self.output_thread = threading.Thread(target=self.consume_output).start()
        return self
    
    def stop(self):
        self.key_queue.put(None)
        for consumer in self.consumers:
            consumer.stop()
        self.input_thread.join()
        self.output_thread.join()

    def consume_input(self):
        input_queue = self.input_queue
        key_queue = self.key_queue
        while True:
            data = input_queue.get()
            if data is None:
                break
            key = self.next_key
            self.next_key += 1
            # find the consumer with the least amount of data
            min_consumer_id = 0
            min_data_count = self.consumers[0].input_queue.qsize()
            for consumer_id, consumer in enumerate(self.consumers):
                if consumer.input_queue.qsize() < min_data_count:
                    min_data_count = consumer.input_queue.qsize()
                    min_consumer_id = consumer_id
            # print(f'Got data {key}, put to min consumer {min_consumer_id}')
            key_queue.put((key, min_consumer_id))
            st_time = time.time()
            self.consumers[min_consumer_id].input_queue.put(data)
            print(f'Put data {key}, FPS: {1/(time.time()-st_time):.2f}')
    
    def consume_output(self):
        output_queue = self.output_queue
        key_queue = self.key_queue
        while True:
            key_consumer_id = key_queue.get()
            if key_consumer_id is None:
                break
            key, consumer_id = key_consumer_id
            # print(f'Wait key {key} from consumer {consumer_id}')
            result = self.consumers[consumer_id].output_queue.get()
            output_queue.put(result)

def main():
    video_path = 'data/HDR80_A_Live_20230211_153630_000.mov'
    video_reader = VideoGear(source=video_path, logging=True).start()
    output_path = Path('temp/output.mp4')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_writer = WriteGear(output=output_path, logging=True)
    model_path = 'yolov8n_mikasa_1280_v1.pt'
    num_process_per_device = 8

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
    devices = ['0', '1']
    for device in devices:
        for i in range(num_process_per_device):
            consumers.append(QueueConsumer(partial(create_consumer, device), postprocess).start())
    pool = QueueConsumerPool(consumers).start()
    print('Start consuming')
    st_time = time.time()
    label_queue = queue.Queue()
    frame_queue = queue.Queue()
    def read_funct():
        frame_count = 0
        while True:
            st_time = time.time()
            frame = video_reader.read()
            # print(f'Read frame {frame_count}, FPS: {frame_count/(time.time()-st_time):.2f}')
            st_time = time.time()
            frame_queue.put(frame)
            # print(f'Put frame {frame_count}, FPS: {frame_count/(time.time()-st_time):.2f}')
            if frame is None:
                break
            frame_count += 1
            st_time = time.time()
            pool.input_queue.put(frame)
            # print(f'Put frame {frame_count}, FPS: {frame_count/(time.time()-st_time):.2f}')
    thread = threading.Thread(target=read_funct)
    thread.start()
    frame_count = 0
    st_time = time.time()
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        results = pool.output_queue.get()
        # video_writer.write()
        # print(results)
        frame_count += 1
        elapsed_time = time.time() - st_time
        print(f'Elapsed time: {elapsed_time:.2f}, FPS: {frame_count/elapsed_time:.2f}, {len(results)} boxes')

    pool.stop()
    thread.join()

if __name__ == '__main__':
    main()
