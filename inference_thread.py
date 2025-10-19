import torch
import torch.version
print(f"torch version: {torch.__version__}")
print(f"torch cuda:  {torch.cuda.is_available()}, {torch.version.cuda}")
print(f"torch cudnn: {torch.backends.cudnn.is_available()}, {torch.backends.cudnn.version()} {torch.backends.cudnn.enabled}, {torch.backends.cudnn.deterministic}")
torch.backends.cudnn.enabled = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.allow_tf32 = False
import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path
import time
from threading import Thread
from queue import Queue
import tqdm
from vidgear.gears import CamGear, VideoGear, WriteGear
import numpy as np


class ThreadOrderedQueue:
    def __init__(self, input_queue) -> None:
        self.input_queue = input_queue
        self.result_dict = {}

    def get(self, wanted_data_id):
        while wanted_data_id not in self.result_dict:
            data = self.input_queue.get()
            data_id, result = data
            self.result_dict[data_id] = result
        return self.result_dict.pop(wanted_data_id)

def read_thread_funct(video_path, output_queue, key_queue):
    # video = cv2.VideoCapture(str(video_path), cv2.CAP_ANY)
    video = VideoGear(source=str(video_path), logging=False).start()
    # print("Using backend:", video.getBackendName())
    frame_id = 0
    batch = []
    while True:
        # ret, frame = video.read()
        frame = video.read()
        frame_id += 1
        # if not ret:
        if frame is None:
            key_queue.put(None)
            output_queue.put(None)
            print("read done")
            break
        key_queue.put(frame_id)
        # frame = letterbox(frame, 1280)
        # frame = ndarray_to_tensor(frame, "cuda:0")
        # print("frame:", frame)
        output_queue.put((frame_id, frame))
        # batch.append((frame_id, frame))
        # if len(batch) == 3:
        #     if output_queue.qsize() == 0:
        #         print("output_queue size is 0, bottleneck in read")
        #     for b in batch:
        #         output_queue.put(b)
        #     batch = []
        # print(f"bsize={len(batch)}, qlen={output_queue.qsize()}")
    # video.release()
    video.stop()

def move_gpu_thread_funct(input_queue, output_queue, done_syncer, model, imgsz, device="0"):
    while True:
        data = input_queue.get()
        if data is None: 
            print("move done")
            input_queue.put(None)
            if done_syncer.get():
                output_queue.put(None)
            break
        data_id, frame = data
        frame = letterbox(frame, imgsz)
        frame = ndarray_to_tensor(frame, f"cuda:{device}")
        # try:
        #     frame = model.predictor.preprocess(frame)
        #     print("use predictor")
        # except Exception as e:
        #     # print(e)
        #     pass
        output_queue.put((data_id, frame))

def process_thread_funct(input_queue, output_queue, done_syncer, model_path, device, imgsz, thread_num=0, is_tensorrt=False, is_torchscript=False):
    # torch.cuda.set_device(device)
    # model.to(device)
    if is_tensorrt:
        torch.cuda.set_device(torch.device(f"cuda:{device}"))
        model = YOLO(str(model_path.with_suffix(".engine")))
    elif is_torchscript:
        model = YOLO(str(model_path.with_suffix(".torchscript")))
        device = torch.device(f"cuda:{device}")
        # model.to(device)
    else:
        model = YOLO(str(model_path))
        device = torch.device(f"cuda:{device}")
        model.to(device)
    print(f"Device {device}, Thread {thread_num} loaded model")
    while True:
        data = input_queue.get()
        if data is None:
            # if done_syncer.get():
            #     print("model all done")
            #     output_queue.put(None)
            # else:
            #     print("model broadcast None")
            #     input_queue.put(None)
            print("model done")
            break
        # if type(data) != list:
        #     data = [data]
        # data_ids = [d[0] for d in data]
        # frames = [d[1] for d in data]
        # print(f"Device {device}, Thread {thread_num}, Frame {data[0]} detected")
        data_id, frame = data
        result = model.predict(frame, verbose=False, imgsz=imgsz, half=False, device=device)[0]
        # print(f"Device {device}, Thread {thread_num}, Frame {data_id} detected")
        output_queue.put((data_id, result))

def letterbox(im, imgsz):
    # imgsz is list [w, h]
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    # print("shape", shape)
    # print("imgsz", imgsz)
    r = min(imgsz[0] / shape[1], imgsz[1] / shape[0])
    # print("r", r)
    # print("r", r)
    # new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r))
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    # print("new_unpad", new_unpad)
    dw = (imgsz[0] - new_unpad[0]) / 2  # width padding
    dh = (imgsz[1] - new_unpad[1]) / 2  # height padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # padding
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return im

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

def main():
    parser = argparse.ArgumentParser(description="Object Detection on Video")
    parser.add_argument("--video", default="data/HDR80_A_Live_20231014_141501_000.mov", type=str, help="path to input video file")
    parser.add_argument("--model", default="./yolov8n_conti_1280_v1.pt", type=str, help="path to input video file")
    parser.add_argument("--imgsz", default="1280", help="model input size")
    parser.add_argument("--outdir", default="detect_results")
    args = parser.parse_args()
    # Load the YOLOv8 model
    model_path = Path(args.model)
    imgsz = [int(args.imgsz),int(args.imgsz)] if ',' not in args.imgsz else [int(x) for x in args.imgsz.split(',')]
    input_queue = Queue()
    output_queue = Queue()
    key_queue = Queue()
    model_threads = []
    ordered_queue = ThreadOrderedQueue(output_queue)
    if not model_path.exists():
        print(f"Model file {model_path} does not exist")
        return
    # Define the video path
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video file {video_path} does not exist")
        return
    
    print("Processing video file: ", video_path)
    is_tensorrt = False
    is_torchscript = False
    if is_tensorrt and not model_path.with_suffix(".engine").exists():
        model = YOLO(str(model_path))
        model.export(format="engine", imgsz=1280, dynamic=False, simplify=True, workspace=32, half=False)
    if is_torchscript and not model_path.with_suffix(".torchscript").exists():
        model = YOLO(str(model_path))
        model.export(format="torchscript", imgsz=1280, optimize=True)
    tensor_queues = []
    move_threads = []
    is_use_move_threading = False
    threads_per_device = 1
    move_thread_per_device = 4
    model_done_syncer = Queue()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        # model = YOLO(str(model_path))
        # model = YOLO(str(model_path.with_suffix(".engine")))
        # model.to(torch.device(f"cuda:{i}"))
        # device = torch.device(f"cuda:{i}")
        device = i
        tensor_queue = Queue(maxsize=30)
        tensor_queues.append(tensor_queue)
        for j in range(threads_per_device):
            print(f"create {j}-th model on device {i}, {device}")
            if is_use_move_threading:
                model_t = Thread(target=process_thread_funct, args=(tensor_queue, output_queue, model_done_syncer, model_path, device, imgsz, j, is_tensorrt, is_torchscript))
            else:
                model_t = Thread(target=process_thread_funct, args=(input_queue, output_queue, model_done_syncer, model_path, device, imgsz, j, is_tensorrt, is_torchscript))
            model_threads.append(model_t)
        if is_use_move_threading:
            move_done_syncer = Queue()
            for k in range(move_thread_per_device - 1):
                move_done_syncer.put(False)
            move_done_syncer.put(True)
            for k in range(move_thread_per_device):
                move_t = Thread(target=move_gpu_thread_funct, args=(input_queue, tensor_queue, move_done_syncer, model, imgsz, i))
                move_threads.append(move_t)
                move_t.start()

    for i in range(1, len(model_threads)):
        model_done_syncer.put(False)
    model_done_syncer.put(True)

    for t in model_threads:
        t.start()
    read_thread = Thread(target=read_thread_funct, args=(video_path, input_queue, key_queue))
    # print(f"Model loaded successfully, class names: {model.names}")
    print(f"Model loaded successfully")
    # Open the video file
    video = cv2.VideoCapture(str(video_path))

    # Get the video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    # Create a VideoWriter object to write the output video
    output_filename = f"{video_path.stem}_out.mp4"
    outputdir = args.outdir if args.outdir is not None else video_path.parent
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    output_path = outputdir / output_filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    output_params = {"-vcodec":"libx264", "-crf":17, "-preset":"medium", "-tune":"zerolatency", "-input_framerate": fps, "-output_dimensions": (width, height)}
    output_video = WriteGear(str(output_path), compression_mode=True, logging=False, **output_params)

    # Process each frame in the video and compute FPS
    sample_count = 0
    start_time = time.time()
    read_thread.start()
    bar = tqdm.tqdm(total=num_frames)
    overall_start_time = time.time()
    frame_id = 0
    while True:
        frame_id += 1
        bar.update(1)
        sample_count += 1
        if sample_count == 30:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = sample_count / elapsed_time
            # print(f"Processed 30 frames in {elapsed_time:.2f} seconds, FPS: {fps:.2f}")
            sample_count = 0
            start_time = time.time()
            # for qidx, q in enumerate(tensor_queues):
            #     print(f"Queue {qidx} size: {q.qsize()}")

        # Perform object detection on the frame
        key = key_queue.get()
        if key is None:
            input_queue.put(None)
            print("key is None")
            break
        result = ordered_queue.get(key)
        frame = result.plot()
        boxes = result.boxes
        # Draw bounding boxes on the frame
        for xywh, class_id, conf in zip(boxes.xywh, boxes.cls, boxes.conf):
            x, y, w, h = xywh.tolist()
            # label = model.names[class_id.item()]
            confidence = conf.item()
            # convert center xywh to xyxy
            x1, y1, x2, y2 = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # frame = result.plot()
        cv2.putText(frame, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        output_video.write(frame)
        # if sample_count == 15:
        #     cv2.imwrite("frame.jpg", frame[..., ::-1])

    print(f"[Overall time]: {time.time() - overall_start_time:.2f}, FPS: {frame_id / (time.time() - overall_start_time):.2f}")
    # Release the video file and the output video
    # print("join threads")
    for t in model_threads:
        t.join()
    for t in move_threads:
        t.join()
    print("threads joined")
    # output_video.release() 
    output_video.close()

    print(f" Output video saved to {output_path}")

if __name__ == "__main__":
    main()