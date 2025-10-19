import cv2 as cv
import argparse
import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", type=str, nargs="+")
    parser.add_argument("--out", type=str)
    args = parser.parse_args()
    if len(args.videos) < 2:
        raise ValueError("Please specify 2 videos")
    cap1 = cv.VideoCapture(args.videos[0])
    cap2 = cv.VideoCapture(args.videos[1])
    if args.out:
        out = cv.VideoWriter(args.out, cv.VideoWriter_fourcc(*'MP4V'), int(cap1.get(cv.CAP_PROP_FPS) + .5), (int(cap1.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT)*2)))
        print(cap1.get(cv.CAP_PROP_FPS), (int(cap1.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))))
    else:
        out = None
    is_same = True
    tqdm_bar = tqdm.tqdm(total=int(cap1.get(cv.CAP_PROP_FRAME_COUNT)))
    while True:
        tqdm_bar.update(1)
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            if not ret1 and ret2:
                print("video 1 ended first")
                is_same = False
            elif ret1 and not ret2:
                print("video 2 ended first")
                is_same = False
            else:
                print("Both video ended")
                is_same = False
            break
        if not (frame1 == frame2).all():
            # print("Frame not equal")
            is_same = False
        if out is not None:
            vf = cv.vconcat([frame1, frame2])
            # print(vf.shape)
            out.write(vf)
    if out is not None:
        out.release()
    print("Same" if is_same else "Not same")

main()