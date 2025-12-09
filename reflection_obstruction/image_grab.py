from enum import Enum
import imageio
from PIL import Image
import numpy as np
from numpy._core.numerictypes import uint8
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import cv2 as cv

video_path = 'images/IMG_1812.mov'
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
print(frames[0])
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
theta = np.atan2(-frame_y, frame_x)
print(theta)
print(theta.min())
print(theta.max())

# def quantize_degrees(theta):
#     theta_return = theta % 180
#     if theta < 22.5:
#         theta_return = 0
#     elif theta < 67.5:
#         theta_return = 45
#     elif theta < 112.5:
#         theta_return = 90
#     else:
#         theta_return = 0

#     return theta_return


theta_deg = ((theta * (180.0/3.141592) + 22.5)% 180) / 45
print(np.floor(theta_deg).max())
rounded_gradients = np.floor(theta_deg)

gradient_image = np.zeros((rounded_gradients.shape[0],rounded_gradients.shape[1], 3), dtype=uint8)

for i, row in enumerate(rounded_gradients):
    for j, pixel in enumerate(row):
        if pixel == 0: #east/west
            gradient_image[i,j] = (255, 255, 0)
        elif pixel == 1: #NE
            # if j % 10 == 0:
                # print(theta[i,j] * (180/3.1415))
            gradient_image[i,j] = (0, 255, 0)
        elif pixel == 2: #N
            gradient_image[i,j] = (0, 0, 255)
        else: #NW
             gradient_image[i,j] = (255, 0, 0)
        # print(pixel)
        # if j > 20:
        #     break
    # if i > 20:
    #     break

pil_img = Image.fromarray(gradient_image)
# pil_img.show()

def get_linear_window(pixel_gradient, i, j, x_bound, y_bound):
    coords = []
    if pixel_gradient == 0:
        for col_idx in range(j-2, j+3):
            if col_idx >= 0 and col_idx < x_bound:
                coords.append((i, col_idx))
    elif pixel_gradient == 1:
        for row_idx, col_idx in zip(range(i+2, i-3, -1),range(j-2, j+3)):
            if (row_idx >= 0 and row_idx < y_bound) and (col_idx >= 0 and col_idx < x_bound):
                coords.append((row_idx, col_idx))
    elif pixel_gradient == 2:
        for row_idx in range(i-2, i+3):
            if row_idx >= 0 and row_idx < y_bound:
                coords.append((row_idx, j))
    elif pixel_gradient == 3:
        for row_idx, col_idx in zip(range(i-2, i+3),range(j-2, j+3)):
            if (row_idx >= 0 and row_idx < y_bound) and (col_idx >= 0 and col_idx < x_bound):
                coords.append((row_idx, col_idx))

    return coords

print(get_linear_window(3, 119, 119, 120, 120))
