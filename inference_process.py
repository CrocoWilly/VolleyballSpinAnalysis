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

class YOLOPredictor:
    def __init__(self, model_path, device='cpu', predict_params=None) -> None:
        self.device = device
        if type(self.device) in [str, int] and self.device != 'cpu':
            self.device = torch.device(f"cuda:{self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.predict_params = predict_params if predict_params is not None else {}

    def predict(self, frame):
        results = self.model.predict(frame, device=self.device, **self.predict_params)
        return results
    
    def __call__(self, frame):
        return self.predict(frame)

class ProcessNumpyConsumer:
    """
        Process manager, use hybrid of multiprocessing and threading to manage the process,
        use thread to pass data_id to make user use any type as data_id (it needs to be hashable)
    """
    def __init__(self, create_consumer, input_queue=None, output_queue=None, max_mbytes=1024) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        if self.input_queue is None:
            self.input_queue = multiprocessing.Queue()
        if self.output_queue is None:
            self.output_queue = multiprocessing.Queue()
        self.process_queue = ArrayQueue(max_mbytes=max_mbytes)
        self.create_consumer = create_consumer
        if not callable(self.create_consumer):
            raise ValueError("create_consumer must be a callable function")
        self.data_id_queue = queue.Queue()
        self.sender_thread = threading.Thread(target=self.sender, args=(self.input_queue, self.data_id_queue, self.process_queue))
        self.receiver_thread = threading.Thread(target=self.receiver, args=(self.data_id_queue, self.output_queue, self.process_queue))
        self.result_table = {}
        
    def start(self):
        self.process = multiprocessing.Process(target=self.consume)
        self.process.start()

    def sender(self, input_queue, output_queue, process_queue):
        while True:
            data = input_queue.get()
            if data is None:
                break
            data_id, process_data = data
            process_queue.put(process_data)
            output_queue.put(data_id)

    def receiver(self, input_queue, output_queue, process_queue):
        while True:
            data = input_queue.get()
            if data is None:
                break
            data_id = data
            process_data = process_queue.get()
            output_queue.put((data_id, process_data))

    def consume(self, input_queue, output_queue):
        self.consumer = self.create_consumer()
        while True:
            data = input_queue.get()
            if data is None:
                break
            data = self.consumer(data)
            output_queue.put(data)

class ProcessResultOrderer:
    def __init__(self, input_queue=None, output_queue=None, queue_maxsize=10) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        if self.input_queue is None:
            self.input_queue = queue.Queue(maxsize=queue_maxsize)
        if self.output_queue is None:
            self.output_queue = queue.Queue(maxsize=queue_maxsize)
        self.result_table = {}
    
    def get(self, target_data_id):
        while target_data_id not in self.result_table:
            data = self.input_queue.get()
            if data is None:  # this should not happended
                continue
            data_id, process_data = data
            self.result_table[data_id] = process_data
        result = self.result_table.pop(target_data_id)
        return result

def main():
    pass