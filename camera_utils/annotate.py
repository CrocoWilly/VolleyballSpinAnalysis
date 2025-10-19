import cv2 as cv
import numpy as np
import argparse
from pathlib import Path
import json
import tqdm

class AnnotateDialog:
    def __init__(self, scale=1.0) -> None:
        self.scale = scale
    
    def open_click_image(self,img , point_cnt=None, window_name="image"):
        points = []
        img_stack = [img]
        scale = self.scale

        def draw_circle(event, x, y, flags, param):
            # if event == cv.EVENT_LBUTTONDBLCLK:
            if event == cv.EVENT_LBUTTONDOWN:
                if point_cnt is not None and len(points) >= point_cnt:
                    return
                # print(x, y)
                new_img = np.array(img_stack[-1])
                cv.circle(new_img, (x, y), 10, (0, 0, 0), 3)
                cv.circle(new_img, (x, y), 0, (0, 0, 255), 1)
                cv.putText(new_img, f"{len(points)+1}({x},{y})", (x, max(0, y - 40)),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv.LINE_AA)
                img_stack.append(new_img)
                points.append([x, y])
                # points.append([x / scale, y / scale])
                cv.imshow(window_name, new_img)
                cv.resizeWindow(window_name, int(new_img.shape[1] * scale), int(new_img.shape[0] * scale))

        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(window_name, draw_circle)
        
        while(1):
            img = img_stack[-1]
            cv.imshow(window_name, img)
            cv.resizeWindow(window_name, int(img.shape[1] * scale), int(img.shape[0] * scale))
            k = cv.waitKey(20) & 0xFF
            if k == 27:
                if point_cnt is not None and len(points) < point_cnt:
                    print(f"Not enough points {len(points)} < {point_cnt}")
                    continue
                break
            elif k == ord('a'):
                if len(img_stack) > 1:
                    print("Revert")
                    img_stack.pop()
                    points.pop()
                pass
        print(points)
        cv.destroyWindow(window_name)
        return points
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=str)
    parser.add_argument("--ext", action="store_true", help="Extract frames from video")
    parser.add_argument("--video", type=str)
    parser.add_argument("--gap", type=int, default=30)

    parser.add_argument("--label", action="store_true")
    parser.add_argument("--img", type=str, help="Manual select which image to use, None to auto select")
    parser.add_argument("--type", type=str, default="main", help="type name of the annotation")
    parser.add_argument("--scale", type=float, default=1.0, help="scale of the image")
    args = parser.parse_args()
    camera_path = Path(args.cam)
    scale = float(args.scale)
    possible_exts = ["mov", "mp4"]
    if args.ext:
        video_path = None
        if args.video:
            video_path = Path(args.video)
        for ext in possible_exts:
            if video_path is None or not video_path.exists():
                video_path = camera_path / f"{camera_path.stem}.{ext}"
            if video_path is None or not video_path.exists():
                video_path = camera_path.parent / f"{camera_path.stem}.{ext}"
        print("video_path", video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        sample_path = camera_path / "samples"
        sample_path.mkdir(exist_ok=True)
        cap = cv.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_id = 0
        tqdm_bar = tqdm.tqdm(total=total_frames)
        while True:
            ret, frame = cap.read()
            frame_id += 1
            tqdm_bar.update(1)
            if not ret:
                break
            if (frame_id - 1) % args.gap != 0:
                continue
            cv.imwrite(str(sample_path / f"{frame_id}.jpg"), frame)
            # print(str(sample_path / f"{frame_id}.jpg"))
    if args.label:
        if not args.type:
            raise ValueError("type should be specified")
        img_path = None
        if args.img:
            img_path = Path(args.img)
        if img_path is None or not img_path.exists():
            img_path = camera_path / "samples" / "0.jpg"
        if img_path is None or not img_path.exists():
            img_path = camera_path / "samples" / "1.jpg"
        if img_path is None or not img_path.exists():
            raise FileNotFoundError(img_path)
        img = cv.imread(str(img_path))
        ref_dir = camera_path / "refs"
        ref_dir.mkdir(exist_ok=True)
        ref_path = ref_dir / (args.type + ".json")
        pts_path = ref_dir / (args.type + "_pts.json")
        if ref_path.exists():
            with open(ref_path, "r") as f:
                ref = json.load(f)
            points_cnt = len(ref)
        else:
            points_cnt = None
        
        dialog = AnnotateDialog(scale=scale)
        points = dialog.open_click_image(img, points_cnt, window_name=f"{pts_path}, #points={points_cnt}")
        print(points)
        with open(pts_path, "w") as f:
            json.dump(points, f)

if __name__ == "__main__":
    main()