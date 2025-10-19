from argparse import ArgumentParser
import cv2 as cv
import numpy as np
from pathlib import Path
import imutils
from PIL import Image, ImageDraw, ImageFont

def draw_text_center(img, text, pos, font_size=20, color=(0, 0, 0)):
    """
    Draw text on the image at the center of the given position.
    :param img: Image to draw on
    :param text: Text to draw
    :param pos: Position to draw the text at (x, y)
    :param font_size: Font size of the text
    :param color: Color of the text (BGR)
    """
    font = ImageFont.truetype("NotoSansTC-Medium.ttf", font_size)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    text_width, text_height = font.getbbox(text)[2:4]
    x = pos[0] - text_width // 2
    y = pos[1] - text_height // 2
    draw.text((x, y), text, font=font, fill=color)
    return np.array(img_pil)

def main():
    parser = ArgumentParser()
    parser.add_argument('gamedir', help='Path to the video file')
    parser.add_argument('--teamimgs', help='Path to the team 1 attack angle image', default="./team_imgs")
    parser.add_argument('--teamnames', help='String of team names', nargs='*', default=None)
    parser.add_argument('--duration', help='Duration of the video', default=30)
    parser.add_argument('--fps', help='FPS of the output video', default=30)
    parser.add_argument('--noteam', action='store_true')
    parser.add_argument('--res', default="1920x1080")
    
    args = parser.parse_args()
    game_dir = Path(args.gamedir)
    team_names = args.teamnames
    video_size_str = args.res
    video_size = [int(_) for _ in video_size_str.split('x')]
    team_1_angle_image_path = game_dir / 'team_1_attack_angle.png'
    team_2_angle_image_path = game_dir / 'team_2_attack_angle.png'
    team_1_arrow_image_path = game_dir / 'team_1_attack_arrow.png'
    team_2_arrow_image_path = game_dir / 'team_2_attack_arrow.png'
    if team_names is not None:
        assert len(team_names) == 2, "team names should be a list of two strings"
        team_imgs_dir = None
        team_1_logo_path = None
        team_2_logo_path = None
    else:
        team_imgs_dir = Path(args.teamimgs)
        team_1_logo_path = team_imgs_dir / 'logo1.png'
        team_2_logo_path = team_imgs_dir / 'logo2.png'
    
    out_path = game_dir / 'analysis.mp4'
    
    num_frames = args.duration * args.fps

    # Generate the video with combined images
    # horizontal stack the logo and image for the team
    # vertical stack the team 1 and team 2 images

    team_1_angle_image = cv.imread(str(team_1_angle_image_path))
    team_2_angle_image = cv.imread(str(team_2_angle_image_path))
    team_1_arrow_image = cv.imread(str(team_1_arrow_image_path))
    team_2_arrow_image = cv.imread(str(team_2_arrow_image_path))
    if team_1_logo_path is not None:
        team_1_logo = cv.imread(str(team_1_logo_path), cv.IMREAD_UNCHANGED)
    else:
        team_1_logo = np.ones((100, 100, 3), dtype=np.uint8) * 255
    if team_2_logo_path is not None:
        team_2_logo = cv.imread(str(team_2_logo_path), cv.IMREAD_UNCHANGED)
    else:
        team_2_logo = np.ones((100, 100, 3), dtype=np.uint8) * 255

    if len(team_1_logo[0][0]) == 4:
        print("Converting BGRA2BGR team 1 logo")
        # make the background white, team_1_logo is 0~255
        alpha = team_1_logo[:, :, 3]
        team_1_logo = team_1_logo[:, :, :3]
        team_1_logo[alpha == 0] = 255
        if args.noteam:
            team_1_logo[:, :, :] = 255

    if len(team_2_logo[0][0]) == 4:
        print("Converting BGRA2BGR team 2 logo")
        alpha = team_2_logo[:, :, 3]
        team_2_logo = team_2_logo[:, :, :3]
        team_2_logo[alpha == 0] = 255
        if args.noteam:
            team_2_logo[:, :, :] = 255
    row_pixels = 800 // 2
    logo_pixels = 200

    # the angle image have large top and bottom margin, crop it 
    team_1_angle_image = team_1_angle_image[100:-100, :]
    team_2_angle_image = team_2_angle_image[100:-100, :]

    team_1_angle_image = imutils.resize(team_1_angle_image, height=row_pixels)
    team_2_angle_image = imutils.resize(team_2_angle_image, height=row_pixels)
    team_1_arrow_image = imutils.resize(team_1_arrow_image, height=row_pixels)
    team_2_arrow_image = imutils.resize(team_2_arrow_image, height=row_pixels)
    
    team_1_logo = imutils.resize(team_1_logo, height=logo_pixels)
    team_2_logo = imutils.resize(team_2_logo, height=logo_pixels)
    # Add margin around to the logo to make it be the same height as the angle image
    padding_row_pixels = (row_pixels - logo_pixels) // 2
    padding_col_pixels = 100
    team_1_logo = cv.copyMakeBorder(team_1_logo, padding_row_pixels, padding_row_pixels, \
                                    padding_col_pixels, padding_col_pixels, cv.BORDER_CONSTANT, value=(255, 255, 255))
    team_2_logo = cv.copyMakeBorder(team_2_logo, padding_row_pixels, padding_row_pixels, \
                                    padding_col_pixels, padding_col_pixels, cv.BORDER_CONSTANT, value=(255, 255, 255))
    
    if team_names is not None:
        team_1_logo = draw_text_center(team_1_logo, team_names[0], (team_1_logo.shape[1] // 2, team_1_logo.shape[0] // 2), font_size=30)
        team_2_logo = draw_text_center(team_2_logo, team_names[1], (team_2_logo.shape[1] // 2, team_2_logo.shape[0] // 2), font_size=30)
    
    team_1_combined = np.hstack([team_1_logo, team_1_angle_image, team_1_arrow_image])
    team_2_combined = np.hstack([team_2_logo, team_2_angle_image, team_2_arrow_image])

    combined = np.vstack([team_1_combined, team_2_combined])
    resized_image = np.ones((video_size[1], video_size[0], combined.shape[2]), dtype=np.uint8) * 255
    margin_h = resized_image.shape[0] - combined.shape[0]
    margin_w = resized_image.shape[1] - combined.shape[1]
    resized_image[margin_h // 2: margin_h // 2 + combined.shape[0], margin_w // 2: margin_w // 2 + combined.shape[1], :] = combined

    # out = cv.VideoWriter(str(out_path), cv.VideoWriter_fourcc(*'mp4v'), args.fps, (combined.shape[1], combined.shape[0]))
    out = cv.VideoWriter(str(out_path), cv.VideoWriter_fourcc(*'mp4v'), args.fps, (video_size[0], video_size[1]))
    for i in range(num_frames):
        # out.write(combined)
        out.write(resized_image)
    out.release()

if __name__ == '__main__':
    main()