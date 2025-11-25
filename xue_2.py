from skimage.registration import phase_cross_correlation
import numpy as np
import imageio
from PIL import Image
import numpy as np
from numpy._core.numerictypes import uint8
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from scipy.ndimage import shift
import cv2 as cv


video_path = 'images/bright_vulture.MOV'
reader = imageio.get_reader(video_path, 'ffmpeg')

frames = []
for i, frame in enumerate(reader):
    if i < 30 and i > 23 and i:
        frames.append(frame)

canny_frames = []
for frame in frames:
    canny_edge = cv.Canny(frame, 100, 200)
    canny_frames.append(canny_edge)


# Getting motion
def get_shift(img1, img2):
    shift, error, diffphase = phase_cross_correlation(
        reference_image=img1,
        moving_image=img2,
        upsample_factor=10
    )
    motion_y, motion_x = shift[:2]
    return motion_y, motion_x

def simple_avg(frames):
    stacked_frame = np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=float)
    for i in range(0, len(frames), 1):
        stacked_frame = np.add(stacked_frame, frames[i])

    stacked_frame = (stacked_frame/len(frames)).astype(uint8)
    pil_img = Image.fromarray(stacked_frame)
    pil_img.show()
    pil_img = Image.fromarray(frames[0])
    pil_img.show()

def min_images(frames, canny_frames, ref_index=len(frames)//2):
    min_frame = np.zeros((frames[0].shape[0], frames[0].shape[1], 3),dtype=uint8) + 255
    for i in range(0, len(frames), 1):
        motion_y, motion_x = get_shift(canny_frames[ref_index], canny_frames[i])
        shifted = shift(frames[i], (motion_y, motion_x, 0))
        min_frame = np.minimum(min_frame, shifted)

    pil_img = Image.fromarray(frames[ref_index])
    pil_img.show()
    pil_img = Image.fromarray(min_frame)
    pil_img.show()



# Stacking images:
def stack_images(frames, canny_frames = None, ref_index=3):
    stacked_frame = np.zeros((frames[0].shape[0], frames[0].shape[1], 3),dtype=float)
    for i in range(0, len(frames), 1):
        if canny_frames == None:
            motion_y, motion_x = get_shift(frames[3], frames[i])
        else:
            motion_y, motion_x = get_shift(canny_frames[3], canny_frames[i])

        shifted = shift(frames[i], (motion_y, motion_x, 0))
        stacked_frame = np.add(stacked_frame, shifted)

    stacked_frame = (stacked_frame/len(frames)).astype(uint8)
    pil_img = Image.fromarray(stacked_frame)
    pil_img.show()
    pil_img = Image.fromarray(frames[3])
    pil_img.show()


# stack_images(frames, canny_frames)
min_images(frames, canny_frames)
# simple_avg(frames)

pil_img = Image.fromarray(canny_frames[3])
pil_img.show()







    # pil_img = Image.fromarray(shifted)
    # pil_img.show()

# new_min_frame = np.zeros((frames[0].shape[0], frames[0].shape[1], 3),dtype=uint8) + 255
# for i in range(0, len(frames), 1):
#     # motion_y, motion_x = get_shift(canny_frames[3], canny_frames[i])
#     motion_y, motion_x = get_shift(np.subtract(frames[3], min_frame), frames[i])

#     shifted = shift(frames[i], (motion_y, motion_x, 0))

#     min_frame = np.minimum(new_min_frame, shifted)
    # stacked_frame = np.add(min_frame, shifted)
    #change shifted to float32:
    # shifted.


    # pil_img = Image.fromarray(shifted)
    # pil_img.show()


# stacked_frame = (stacked_frame/len(frames)).astype(uint8)
# pil_img = Image.fromarray(stacked_frame)
# pil_img.show()
# #
# pil_img = Image.fromarray(frames[0])
# pil_img.show()
# pil_img = Image.fromarray(min_frame)
# pil_img.show()
