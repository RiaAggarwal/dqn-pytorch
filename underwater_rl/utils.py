import glob
import os
import re

import cv2


def convert_images_to_video(image_dir: str, save_dir: str):
    """
    Converts the images in image_dir to an mp4; deletes the original images
    :param image_dir:
    :return:
    """
    assert os.path.isdir(image_dir), f"Directory {image_dir} does not exist"

    filenames = [f for f in glob.glob(os.path.join(image_dir, '*.png'))]
    filenames.sort(key=lambda f: int(re.findall(r'\/(\d+).png', f).pop()))

    img_array = []
    for f in filenames:
        img = cv2.imread(f)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(os.path.join(save_dir, 'pong.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
