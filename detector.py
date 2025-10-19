import threading
from ultralytics import YOLO
from queue import Queue


class CameraSetDetector:
    def __init__(self, detectors) -> None:
        self.results = {}
        self.result_queue = Queue()  # just for blocking
        self.queue = Queue()  # for consumer
        self.get_queue = Queue()  # for getter knows the target
        self.detectors = detectors
        self.detector_threads = []
        for detector in self.detectors:
            t = threading.Thread(target=self.run_detector, args=(detector,))
            t.start()
            self.detector_threads.append(t)

    def run_detector(self, detector):
        while True:
            data = self.queue.get()
            if data is None:
                self.result_queue.put(None)
                break
            cam, frame = data
            result = detector.predict(frame, verbose=False, imgsz=1280, half=False)[0]
            self.results[(cam, frame)] = result
            self.result_queue.put((cam, frame, result))

    def input(self, camera_frames):
        for cam, frame in camera_frames.items():
            self.queue.put((cam, frame))
            self.get_queue.put((cam, frame))

    def get(self):
        target = self.get_queue.get()
        while True:
            if target in self.results:
                result = self.results[target]    
                del self.results[target]
                return result
            result = self.result_queue.get()
        
        