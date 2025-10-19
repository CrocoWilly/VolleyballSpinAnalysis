import threading
from queue import Queue
import multiprocessing
import time
from loguru import logger
import numpy as np


class PipelineStage:
    # class for wrapping ugly sentinel operations for stages
    def __init__(self, funct, input_queue=None, output_queue=None, queue_max_size=10, on_finished=None) -> None:
        self.input_queue = Queue(maxsize=queue_max_size) if input_queue is None else input_queue
        self.output_queue = Queue(maxsize=queue_max_size) if output_queue is None else output_queue
        self.funct = funct
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        self.on_finished = on_finished

    def run(self):
        while True:
            data = self.input_queue.get()
            if data is None:
                self.output_queue.put(None)
                break
            result = self.funct(data)
            self.output_queue.put(result)
        if self.on_finished:
            self.on_finished()

class ProcessPipelineStage:
    def __init__(self, funct, input_queue=None, output_queue=None, queue_max_size=10, process_name=None, context_ctor=None) -> None:
        self.input_queue = multiprocessing.Queue(maxsize=queue_max_size) if input_queue is None else input_queue
        self.output_queue = multiprocessing.Queue(maxsize=queue_max_size) if output_queue is None else output_queue
        self.funct = funct
        self.process_name = process_name
        self.context_ctor = context_ctor
        self.process = multiprocessing.Process(target=self.run, args=(self.funct, self.input_queue, self.output_queue, self.process_name, self.context_ctor))
        self.process.start()

    @staticmethod
    def run(funct, input_queue, output_queue, process_name, context_ctor):
        if context_ctor:
            context = context_ctor()
        else:
            context = None
        while True:
            data = input_queue.get()
            if data is None:
                output_queue.put(None)
                break
            st_time = time.time()
            if context is not None:
                result = funct(data, context)
            else:
                result = funct(data)
            # logger.debug(f"Process {funct.__name__} took {time.time() - st_time:.2f} s (FPS: {1 / (time.time() - st_time):.2f})")
            logger.debug(f"Process {process_name} took {time.time() - st_time:.2f} s (FPS: {1 / (time.time() - st_time):.2f})")
            output_queue.put(result)
        logger.debug(f"Process {process_name} finished")

class NullQueue:
    def put(self, data):
        pass

class NumpyQueue:
    def __init__(self, maxsize) -> None:
        self.shared_mem = None
        self.share_arr = None
        self.maxsize = maxsize
        self.dtype = None
        self.current_get_idx = 0
        self.idx_queue = multiprocessing.Queue(maxsize=maxsize)
        for i in range(maxsize):
            self.idx_queue.put(i)

    def setup(self, sample_array):
        data_size = sample_array.nbytes
        self.dtype = sample_array.dtype
        self.shared_mem = multiprocessing.Array('b', data_size * self.maxsize)
        self.shared_np_array = np.frombuffer(self.shared_mem, dtype=self.dtype, count=self.maxsize)
        self.shared_np_array.shape = (self.maxsize, *sample_array.shape)

    def put(self, data):
        if self.shared_mem is None:
            self.setup(data)
        put_idx = self.idx_queue.get()
        self.shared_np_array[put_idx] = data