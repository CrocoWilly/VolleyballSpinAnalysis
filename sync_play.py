import cv2
import numpy as np
import threading
import time
import argparse
import os
import multiprocessing


# 解析命令列參數
parser = argparse.ArgumentParser(description="Sync playback of multiple videos on specified screens.")
parser.add_argument("video_paths", nargs='+', help="List of video file paths.")
parser.add_argument("--screens", nargs='+', type=int, help="List of X screen indices corresponding to each video.")
args = parser.parse_args()

video_paths = args.video_paths
num_videos = len(video_paths)

# 讀取所有影片
caps = [cv2.VideoCapture(path) for path in video_paths]

# 檢查影片的 FPS，確保正確播放
fps_list = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
fps = None
for i, fps in enumerate(fps_list):
    print(f"Video {i+1} FPS: {fps}")

# 設定 FPS
TARGET_FPS = fps
FRAME_TIME = 1.0 / TARGET_FPS

# 確保螢幕數量與影片數量相符
if args.screens and len(args.screens) != num_videos:
    raise ValueError("Number of screens must match the number of video paths.")

# 建立 Barrier，確保所有視角同步播放
# barrier = threading.Barrier(num_videos)
barrier = multiprocessing.Barrier(num_videos)

def show_video(video_path, window_name, thread_idx, screen_index=None):
    # 設定播放的螢幕
    # if screen_index is not None:
    #     os.environ['DISPLAY'] = f":{screen_index}"
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # 現在把螢幕橫向排列，一個螢幕是1920x1080
    cv2.moveWindow(window_name, 1920 * screen_index, 0)

    adjust_wait_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            barrier.abort()
            break
        
        start_time = time.time()
        cv2.imshow(window_name, frame)
        
        # 讓執行緒在這裡等待，確保所有視角同步播放
        try:
            bid = barrier.wait()
        except threading.BrokenBarrierError:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            barrier.abort()
            break
        
        # 保持 60 FPS 播放
        if thread_idx == 0:
            elapsed_time = time.time() - start_time
            sleep_time = FRAME_TIME - elapsed_time
            sleep_time += adjust_wait_time
            sleep_time = max(0, sleep_time)
            time.sleep(sleep_time)
            actual_elapsed_time = time.time() - start_time
            ratio = 0.1
            adjust_wait_time = adjust_wait_time * (1 - ratio) +  (FRAME_TIME - actual_elapsed_time) * ratio
            print("Actual FPS:", 1.0 / (actual_elapsed_time + 1e-6), " Adjust time:", adjust_wait_time)

# 使用多執行緒確保所有視角同步播放
threads = []
for i in range(num_videos):
    screen_index = args.screens[i] if args.screens else None
    # thread = threading.Thread(target=show_video, args=(video_paths[i], f"View{i+1}", i, screen_index))
    thread = multiprocessing.Process(target=show_video, args=(video_paths[i], f"View{i+1}", i, screen_index))
    threads.append(thread)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

for cap in caps:
    cap.release()
cv2.destroyAllWindows()