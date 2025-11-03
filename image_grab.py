import imageio
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import cv2 as cv

video_path = 'images/IMG_1648.mov'
reader = imageio.get_reader(video_path, 'ffmpeg')

frames = []
for i, frame in enumerate(reader):
    if i % 10 == 0:
        frames.append(frame)
        # pil_img = Image.fromarray(frame)
        # pil_img.show()


def normalize(image):
    return_image = ((image - image.min())/(image.max() - image.min()) * 255).astype(np.uint8)
    return return_image


frame1 = rgb2gray(frames[0])
print(len(frames))
# pil_img = Image.fromarray(frames[0])
# pil_img.show()
# print(frames[0])
frame1 = gaussian_filter(frame1, 3, radius=2)
pil_img = Image.fromarray(normalize(frame1))
# print(frames[0])
# pil_img.show()


# frame1 = (frame1 * 255).astype(np.uint8)
# pil_img = Image.fromarray(frame1)
# pil_img.show()
print(frame1.shape)
print(frame1)



frame_y = convolve(frame1, weights=[[1,2,1], [0,0,0], [-1,-2,-1]])
# frame_y = frame_y + frame_y.min()
# frame_y = frame_y/frame_y.max()
print(frame_y.max())
pil_img = Image.fromarray(normalize(frame_y))
# pil_img.show()


frame_x = convolve(frame1, weights=[[1,0,-1], [2,0,-2], [1,0,-1]])
print(frame_x.min())
# frame_x = frame_x/frame_x.max()
# pil_img = Image.fromarray((np.abs(frame_x) * 255).astype(np.uint8))
pil_img = Image.fromarray(normalize(frame_x))
# pil_img.show()

# res_u8 = (np.abs(frame_y) * 255).astype(np.uint8)
# pil_img = Image.fromarray(res_u8)
# pil_img.show()

result = np.hypot(frame_x, frame_y)
# scaled = ((result - result.min())/(result.max() - result.min()) * 255).astype(np.uint8)
# res_u8 = (result * 255).astype(np.uint8)
# res_u8 = cv.normalize(result, 0, 255)

# pil_img = Image.fromarray(res_u8)
# pil_img.show()

pil_img = Image.fromarray(normalize(result))
# pil_img.show()
# print(result)
#

# Normalized to remove negative angle values
# theta = np.atan2(normalize(frame_y), normalize(frame_x))
theta = np.atan2(frame_y, frame_x)
print(theta)
print(theta.min())
print(theta.max())
