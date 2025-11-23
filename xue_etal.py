import imageio
from PIL import Image
import numpy as np
from numpy._core.numerictypes import uint8
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import cv2 as cv


video_path = 'images/allens.mp4'
reader = imageio.get_reader(video_path, 'ffmpeg')

frames = []
for i, frame in enumerate(reader):
    if i % 10 == 0:
        frames.append(frame)

canny_frames = []
for frame in frames:
    canny_edge = cv.Canny(frame, 50, 100)
    # canny_float = cv.normalize(canny_edge, None, 0.0, 1.0, cv.NORM_MINMAX, cv.CV_32F)
    canny_frames.append(canny_edge)

def edge_flow (canny1, canny2):
    y_pos, x_pos = np.where(canny1 > 0)

    point_list = np.float32(np.column_stack((x_pos, y_pos))).reshape(-1,1,2)

    results, status, _ = cv.calcOpticalFlowPyrLK(canny1, canny2, point_list, None)

    good_results = results[status == 1]
    good_points = point_list[status == 1]

    return good_results, good_points

def edge_flow_global(canny1, canny2):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    return cv.findTransformECC(canny1, canny2, warp_matrix, cv.MOTION_AFFINE)

corr, matrix = edge_flow_global(canny_frames[0], canny_frames[3])

output_images = []
for image in canny_frames:

outputImage = cv.warpAffine(frames[3], matrix, (frames[0].shape[1], frames[0].shape[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

print(edge_flow_global(canny_frames[0], canny_frames[3]))


pil_img = Image.fromarray(outputImage)
pil_img.show()
# ransac_res = cv.findHomography(point_list, results, cv.RANSAC, 5.0)
# print(ransac_res)

#Gonna use calcOpticalFlowPyrLK() on these edge detected frames -> ransac?


# print(len(frames))
# print(edge_flow(frame[0], frame[1]))
# pil_img = Image.fromarray(canny_frame)
# pil_img.show()
