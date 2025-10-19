import numpy as np
import time

class ObserveVariable:
    def __init__(self, smooth=None, reset_threshold=10, skip=0, max_record_len=30) -> None:
        self.data = []
        self.frame_ids = []
        self.value = None
        self.frame_id = None
        self.derivative = None
        self.derivative_frame_id = None
        self.smooth = smooth
        self.skip = skip
        self.reset_threshold = reset_threshold
        self.max_record_len = max_record_len

    def reset(self):
        self.data = []
        self.frame_ids = []
        self.value = None
        self.frame_id = None
        self.derivative = None
        self.derivative_frame_id = None

    def input(self, frame_id, value):
        if value is None:
            return None, None
        if self.frame_id is not None and (frame_id - self.frame_id > self.reset_threshold):
            self.reset()
        value = np.array(value, dtype=np.float32)
        self.data.append(value)
        self.frame_ids.append(frame_id)
        if len(self.data) > self.max_record_len:
            self.data.pop(0)
            self.frame_ids.pop(0)
        if len(self.data) >= 2 + self.skip:
            prev_idx = -2 - self.skip
            frame_diff = self.frame_ids[-1] - self.frame_ids[prev_idx]
            new_derivative = (self.data[-1] - self.data[prev_idx]) / frame_diff
            if self.smooth and self.derivative is not None:
                new_derivative = self.derivative * self.smooth + new_derivative * (1 - self.smooth)
            self.derivative = new_derivative
            self.derivative_frame_id = self.frame_ids[prev_idx] + frame_diff / 2
        d = self.derivative.copy() if self.derivative is not None else None
        return d, self.derivative_frame_id
        
class PerfTimer:
    def __init__(self) -> None:
        self.accumualted_time = 0

    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.accumualted_time += time.time() - self.start_time

    def reset(self):
        self.accumualted_time = 0

    @property
    def seconds(self):
        return self.accumualted_time
