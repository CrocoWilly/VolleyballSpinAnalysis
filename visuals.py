import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

def draw_volleyball_court(ax):
    ax: plt.Axes
    points = np.array([
        [0, 0, 0], [9, 0, 0], [0, 6, 0], [9, 6, 0], [0, 9, 0],
        [9, 9, 0], [0, 12, 0], [9, 12, 0], [0, 18, 0], [9, 18, 0]
    ])
    courtedge = [2, 0, 1, 3, 2, 4, 5, 3, 5, 7, 6, 4, 6, 8, 9, 7]
    curves = points[courtedge]

    netpoints = np.array([
        [0, 9, 0], [0, 9, 1.24], [0, 9, 2.24], [9, 9, 0], [9, 9, 1.24], [9, 9, 2.24]])
    netedge = [0, 1, 2, 5, 4, 1, 4, 3]
    netcurves = netpoints[netedge]

    court = points.T
    courtX, courtY, courtZ = court
    # plot 3D court reference points

    ax.scatter(courtX, courtY, courtZ, c='black', marker='o', s=1)
    ax.plot(curves[:, 0], curves[:, 1], c='k',
            linewidth=2, alpha=0.5)  # plot 3D court edges
    ax.plot(netcurves[:, 0], netcurves[:, 1], netcurves[:, 2],
            c='k', linewidth=2, alpha=0.5)  # plot 3D net edges
    
def court_fig(dpi=None):
    fig = plt.figure(dpi=dpi)
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(left=0, right=9)
    ax.set_ylim3d(bottom=-1, top=19)
    # ax.set_zlim3d(bottom=0, top=5)
    ax.set_zlim3d(bottom=0, top=9)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    draw_volleyball_court(ax)

    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.patch.set_facecolor('none')
    ax.patch.set_alpha(0.0)
    pane_color = (0.8, 0.8, 0.8, 1.0)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    axis_line_color = (0.9, 0.9, 0.9, 0.0)
    ax.xaxis.line.set_color(axis_line_color)
    ax.yaxis.line.set_color(axis_line_color)
    ax.zaxis.line.set_color(axis_line_color)

    ax.grid(False)

    return fig, ax

class Court3DFigure:
    def __init__(self, elev=5, azim=195, dpi=None) -> None:
        fig, ax = court_fig(dpi=dpi)
        ax.view_init(elev=elev, azim=azim)
        # Set patches(background) to transparent
        # ax.axis('square')

        self.fig, self.ax = fig, ax
        width, height = fig.get_size_inches() * fig.get_dpi()
        self.width, self.height = int(width), int(height)

        self.pos3d_list = None
        self.color_list = []
        self.canvas = fig.canvas
        self.canvas.draw()

        self.background = self.canvas.copy_from_bbox(ax.bbox)
        self.artist = None
        self.previous_pos3d = None

    def split_pos3d_by_colors(self, pos3d_list, color_list):
        # convert pos3d_list, color to pos3d_seg, color_seg
        if type(color_list) == str:
            return [pos3d_list], [color_list]

        pos3d_segments = []
        segment_color_list = []

        prev_color = None
        seg = None
        for pos3d, color in zip(pos3d_list, color_list):

            if color != prev_color:
                if seg is not None:
                    pos3d_segments.append(seg)
                    seg = None

                if prev_color is None:
                    prev_color = color
                segment_color_list.append(prev_color)
                prev_color = color
            else:
                if seg is None:
                    seg = np.array([pos3d])
                else:
                    seg = np.concatenate([seg, [pos3d]], axis=0)
        return pos3d_segments, segment_color_list

    def add_pos3d(self, pos3d, color='b', with_line=False, s=5):
        self.color_list.append(color)
        if self.pos3d_list is not None:
            self.pos3d_list = np.concatenate(
                [self.pos3d_list, [pos3d]], axis=0)
        else:
            self.pos3d_list = np.array([pos3d])

        fig, ax = self.fig, self.ax
        canvas = self.canvas
        # artist = ax.scatter(*pos3d, s=3, c='b', marker='o')
        # canvas.draw()
        canvas.restore_region(self.background)

        if not with_line:
            # Alt 1 Scatter points
            # print(self.pos3d_list.T)
            artist = ax.scatter(*(self.pos3d_list.T.tolist()),
                                s=s, c=self.color_list, marker='o')
            artist.do_3d_projection()
            ax.draw_artist(artist)
        else:
            # Alt 2　Plot point with line
            pos3d_segs, seg_color_list = self.split_pos3d_by_colors(
                self.pos3d_list, self.color_list)

            for pos3d_list, seg_color in zip(pos3d_segs, seg_color_list):
                print("pos3d_list:", pos3d_list.shape)
                print(seg_color)

        fig.canvas.blit(fig.bbox)

    def get_court_image(self):
        fig, ax = self.fig, self.ax
        canvas, width, height = self.canvas, self.width, self.height
        # canvas.draw()
        court_image = np.frombuffer(canvas.tostring_argb()
                                    , dtype='uint8').reshape(int(height), int(width), 4)
        court_image = court_image[:, :, [1, 2, 3, 0]]
        court_image = cv.cvtColor(court_image, cv.COLOR_RGBA2BGRA)
        # court_image = np.asarray(canvas.buffer_rgba())
        # court_image = cv.cvtColor(court_image, cv.COLOR_RGBA2BGRA)
        crop_xxyy = (160, 490, 120, 355)  # 240x400 in old version
        # print(court_image.shape)
        court_image = court_image[crop_xxyy[2]
            :crop_xxyy[3], crop_xxyy[0]:crop_xxyy[1]]
        return court_image


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    # This will write the background in-place
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]  # .astype(np.float32)
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]  # .astype(np.float32)

    imf, imb = Image.fromarray(foreground.astype(np.uint8)), Image.fromarray(background_subsection.astype(np.uint8))
    imb.putalpha(255)
    composite = Image.alpha_composite(imb, imf).convert("RGB")
    background[bg_y:bg_y + h, bg_x:bg_x + w] = np.asarray(composite)

def paste_image(background, foreground, x_offset=None, y_offset=None):
    # This will write the background in-place
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]  # .astype(np.float32)
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]  # .astype(np.float32)

    # imf, imb = Image.fromarray(foreground.astype(np.uint8)), Image.fromarray(background_subsection.astype(np.uint8))
    # imb.putalpha(255)
    # composite = Image.alpha_composite(imb, imf).convert("RGB")
    composite = foreground[:, :, :3]
    background[bg_y:bg_y + h, bg_x:bg_x + w] = np.asarray(composite)

def draw_line(img, line, color, thickness=3, line_type=cv.LINE_AA):
    # line is in the form of [a, b, c] where a*x + b*y + c = 0
    h, w, _ = img.shape
    a, b, c = line
    if abs(a) > abs(b):
        y0 = 0
        y1 = h
        x0 = int((-c - b*y0) / a)
        x1 = int((-c - b*y1) / a)
    else:
        x0 = 0
        x1 = w
        y0 = int((-c - a*x0) / b)
        y1 = int((-c - a*x1) / b)
    img = cv.line(img, (x0, y0), (x1, y1), color, thickness, line_type)
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv.rectangle(img, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv.rectangle(img, c1, c2, color, -1, cv.LINE_AA)  # filled
        cv.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                   [225, 255, 255], thickness=tf, lineType=cv.LINE_AA)

class Court2DFigure:
    def __init__(self, image_size=(600, 1200)):
        self.court_orig = (-0.5, -1)
        self.court_size = (10, 20)
        self.image_size = image_size
        self.pos3d_list = []
        self.color_list = []
        self.serve_pos3d_list = []
        self.image_cache = None
        self.new_court_image = self.get_new_court_image()

    def add_serve_pos3d(self, start_pos3d, end_pos3d):
        self.image_cache = None
        self.serve_pos3d_list.append((start_pos3d, end_pos3d))
    
    def add_pos3d(self, pos3d, color='b', with_line=False, s=5):
        self.image_cache = None
        self.color_list.append(color)
        self.pos3d_list.append(pos3d)

    def court_pt_to_image_pt(self, pt):
        # y is reversed
        if len(pt) > 2:
            pt = pt[:2]
        img_pt = (pt - self.court_orig) / self.court_size * self.image_size
        img_pt[1] = self.image_size[1] - img_pt[1]
        return img_pt

    def get_new_court_image(self):
        ground_color = (255, 114, 71)
        court_color = (3, 186, 252)
        image = np.full((self.image_size[1], self.image_size[0], 3), 255, dtype=np.uint8)
        image[:, :] = ground_color
        court_left_top = self.court_pt_to_image_pt(np.array([0, 0, 0])).astype(np.uint32)
        court_right_bottom = self.court_pt_to_image_pt(np.array([9, 18, 0])).astype(np.uint32)
        # logger.info(f"court_left_top:{court_left_top}, court_right_bottom:{court_right_bottom}")
        image[int(court_right_bottom[1]):int(court_left_top[1]),  # because y is reversed
              int(court_left_top[0]):int(court_right_bottom[0])] = court_color
        points = np.array([
        [0, 0, 0], [9, 0, 0], [0, 6, 0], [9, 6, 0], [0, 9, 0],
        [9, 9, 0], [0, 12, 0], [9, 12, 0], [0, 18, 0], [9, 18, 0]
        ])
        courtedge = [2, 0, 1, 3, 2, 4, 5, 3, 5, 7, 6, 4, 6, 8, 9, 7]
        courtedge_pts = points[courtedge]

        netpoints = np.array([
            [0, 9, 0], [0, 9, 1.24], [0, 9, 2.24], [9, 9, 0], [9, 9, 1.24], [9, 9, 2.24]])
        netedge = [0, 1, 2, 5, 4, 1, 4, 3]
        netedge_pts = netpoints[netedge]
        court_line_color = (255, 255, 255)
        for i in range(len(courtedge_pts) - 1):
            pt1, pt2 = courtedge_pts[i], courtedge_pts[i + 1]
            pt1 = self.court_pt_to_image_pt(pt1).astype(np.uint32)
            pt2 = self.court_pt_to_image_pt(pt2).astype(np.uint32)
            # logger.info(f"Draw start_pt:{pt1}, end_pt:{pt2}")
            cv.line(image, pt1, pt2, court_line_color, 2)
        for i in range(len(netedge_pts) - 1):
            pt1, pt2 = netedge_pts[i], netedge_pts[i + 1]
            pt1 = self.court_pt_to_image_pt(pt1).astype(np.uint32)
            pt2 = self.court_pt_to_image_pt(pt2).astype(np.uint32)
            cv.line(image, pt1, pt2, court_line_color, 2)
        return image
    
    def get_court_image(self):
        if self.image_cache is not None:
            return self.image_cache
        image = self.new_court_image.copy()
        for pos3d, color in zip(self.pos3d_list, self.color_list):
            x, y, z = pos3d
            if 0 <= x <= 9 and 0 <= y <= 18:
                x = int(x / 9 * self.image_size[1])
                y = int(y / 18 * self.image_size[0])
                try:
                    cv.circle(image, (x, y), 5, (0, 0, 0), -1)
                except Exception as e:
                    logger.error(f"Error in drawing x:{x}, y:{y}, image_size:{self.image_size}")
        for start_pos3d, end_pos3d in self.serve_pos3d_list:
            start_pt = self.court_pt_to_image_pt(start_pos3d).astype(np.uint32)
            end_pt = self.court_pt_to_image_pt(end_pos3d).astype(np.uint32)
            start_color = (61, 61, 255)
            end_color = (56, 219, 44)
            try:
                cv.circle(image, start_pt, 10, start_color, -1)
            except Exception as e:
                logger.error(f"Error in drawing point:{start_pt}, image_size:{self.image_size}")
            try:
                cv.circle(image, end_pt, 10, end_color, -1)
            except Exception as e:
                logger.error(f"Error in drawing point:{end_pt}, image_size:{self.image_size}")
            try:
                cv.arrowedLine(image, start_pt, end_pt, start_color, 5, tipLength=0.05, line_type=cv.LINE_AA)
            except Exception as e:
                logger.error(f"Error in drawing line from {start_pt} to {end_pt}, image_size:{self.image_size}")
        self.image_cache = image
        return image
    
IFfont_size=50
IFfont = None
title_image_cache = {}
def draw_result_info_card(image, xy, title, result, box_color=(255, 255, 255), text_color=(0, 0, 0), is_eng=True):
    global IFfont, title_image_cache
    if IFfont is None:
        st_time = time.time()
        IFfont = ImageFont.truetype("NotoSansTC-Medium.ttf", IFfont_size)
        print(f"time cost for loading font: {time.time() - st_time}")

    # draw a card with title and result directly on the image, need to be fast
    h, w, _ = image.shape
    card_h, card_w = 80, 520
    x, y = xy
    # cv.rectangle(image, (x, y), (x + card_w, y + card_h), box_color, -1, cv.LINE_AA)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    x_margin = 15
    if is_eng:
        image[y:y + card_h, x:x + card_w] = box_color
        title_size, baseline = cv.getTextSize(title, font, font_scale, thickness)
        title_w, title_h = title_size
        text_x = x + x_margin
        text_y = y + card_h // 2 + title_h // 2
        cv.putText(image, title, (text_x, text_y), font, font_scale, text_color, thickness, cv.LINE_AA)

    else:
        if title in title_image_cache:
            title_image = title_image_cache[title]
        else:
            title_image = np.ones((card_h, card_w, len(box_color)), dtype=np.uint8)
            title_image[:,:,] = box_color
            text_width, text_height = IFfont.getbbox(title)[2:4]
            title_w, title_h = text_width, text_height
            # get the text box and draw in center of height
            text_x = x_margin
            text_y = card_h // 2 + title_h // 2
            text_y -= 7  
            print(f"title_image shape {title_image.shape}")
            img_pil = Image.fromarray(title_image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((text_x + 50, text_y - (title_h)), title, font=IFfont, fill=text_color)
            title_image = np.array(img_pil)
            title_image_cache[title] = title_image
        image[y:y + card_h, x:x + card_w] = title_image

    # text_width, text_height = IFfont.getbbox(title)[2:4]
    # title_w, title_h = text_width, text_height

    # print(f"Origin title_size:{title_size}, New:{(text_width, text_height)}")
    # result_size, baseline = cv.getTextSize(result, font, font_scale, thickness)
    # result_w, result_h = result_size
    # x_margin = 15
    # text_x = x + x_margin
    # text_y = y + card_h // 2 + title_h // 2
    # cv.putText(image, title, (text_x, text_y), font, font_scale, text_color, thickness, cv.LINE_AA)
    # text_y -= 7
    # st_time = time.time()
    # img_pil = Image.fromarray(image)
    # draw = ImageDraw.Draw(img_pil)
    # draw.text((text_x + 50, text_y - (title_h)), title, font=IFfont, fill=text_color)
    # image = np.array(img_pil)
    # print(f"Time cost for drawing title: {time.time() - st_time}")

    result_size, baseline = cv.getTextSize(result, font, font_scale, thickness)
    result_w, result_h = result_size
    text_x = x + card_w - x_margin - result_w
    text_y = y + card_h // 2 + result_h // 2
    cv.putText(image, result, (text_x, text_y), font, font_scale, text_color, thickness + 2, cv.LINE_AA)
    # cv.imwrite("result_card.png", image)
    return image

def main():
    # Save the scene as a .png file
    st_time = time.time()
    fig = Court3DFigure(elev=5, azim=195, dpi=100)
    pt = np.array([4.5, 9, 1])
    color = (0, 0, 1)
    for i in range(1000):
        # randomly move a 3D point in the court which 0 < x < 9, 0 < y < 18, 0 < z < 5
        # change the color with small probability
        delta = np.random.randn(3) * [0.05, 0.1, 0.02]
        pt += delta
        pt[0] = np.clip(pt[0], 0, 9)
        pt[1] = np.clip(pt[1], 0, 18)
        pt[2] = np.clip(pt[2], 0, 5)
        if np.random.rand() < 0.02:
            color = np.random.rand(3)
        fig.add_pos3d(pt, color=color, with_line=False, s=5)
        img = fig.get_court_image()
    img = fig.get_court_image()
    cv.imwrite("vispy_test.png", img)
    time_cost = time.time() - st_time
    print("Time:", time_cost)
    print("FPS:", 1000 / time_cost)


if __name__ == '__main__':
    main()
