import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def save_image(path, image):
    cv2.imwrite(path, image)
    cv2.waitKey(10)


def get_image_path_list(data_dir_path):
    return [str(path) for path in Path(data_dir_path).glob("*.jpg")]


def get_mask_path_list(data_dir_path):
    return [str(path) for path in Path(data_dir_path).glob("*.png")]


def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr2hsv(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    return image_hsv


def bgr2lab(image):
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return image_lab


def bgr2blight(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image_hsv[:, :, 1]


def equalize_hist(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def cv2pil(image_array: np.ndarray) -> Image:
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_array)


def resize_int(value, resize_rate: float):
    return int(value * resize_rate)


def rotate_anticlockwise_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def show_image_from_ndarray(image_array):
    if (len(image_array.shape) < 3) or (image_array.shape[-1] == 1):
        plt.imshow(image_array)
        plt.axis("off")
    else:
        plt.imshow(cv2pil(image_array))
        plt.axis("off")


def draw_multiple_image(titles, images, enable_bgr2rgb_mode=False):
    n_images = len(images)
    assert len(titles) == n_images

    fig, axes = plt.subplots(1, n_images)
    for i in range(n_images):
        draw_image = bgr2rgb(images[i]) if enable_bgr2rgb_mode else images[i]
        axes[i].imshow(draw_image)
        axes[i].set_title(titles[i])
        axes[i].axis("off")