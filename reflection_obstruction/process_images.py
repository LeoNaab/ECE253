import os
from datetime import datetime

import cv2 as cv
import imageio
import PIL.Image as Image
import xue_2


def process(method="xue"):
    time = datetime.now().isoformat(timespec="seconds")
    os.makedirs(f"dataset/{method}{time}/")
    for video_file in os.listdir("dataset/unprocessed"):
        if video_file.startswith("."):  # frickin .DS_store
            continue

        video_path = f"dataset/unprocessed/{video_file}"
        reader = imageio.get_reader(video_path, "ffmpeg")

        video_name = video_file.split(".")[0]

        frames = []
        for i, frame in enumerate(reader):
            if i < 40 and i > 15 and i % 2:
                frames.append(frame)

        if method == "xue":
            canny_frames = []
            for frame in frames:
                canny_edge = cv.Canny(frame, 100, 200)
                canny_frames.append(canny_edge)
            print(len(canny_frames))
            print(len(frames))
            processed_image = xue_2.min_images(
                frames=frames, canny_frames=canny_frames, ref_index=len(frames) // 2
            )
            pil_img = Image.fromarray(processed_image)

            pil_img.save(f"dataset/{method}{time}/{video_name}.jpg", format="JPEG")

        elif method == "average":
            processed_image = xue_2.simple_avg(frames)

            pil_img = Image.fromarray(processed_image)
            pil_img.save(f"dataset/{method}{time}/{video_name}.jpg", format="JPEG")

        elif method == "blur":
            processed_image = xue_2.blur_image(frames[len(frames) // 2])

            pil_img = Image.fromarray(processed_image)
            pil_img.save(f"dataset/{method}{time}/{video_name}.jpg", format="JPEG")

        else:
            pil_img = Image.fromarray(frames[len(frames) // 2])
            pil_img.save(f"dataset/{method}{time}/{video_name}.jpg", format="JPEG")


process("blur")
# process("unprocessed_images")
