import os
from datetime import datetime

import cv2 as cv
import imageio
import PIL.Image as Image


def gather():
    time = datetime.now().isoformat(timespec="seconds")

    for video_file in os.listdir("../reflection_obstruction/dataset/unprocessed/"):
        if video_file.startswith("."):  # frickin .DS_store
            continue

        video_path = f"../reflection_obstruction/dataset/unprocessed/{video_file}"
        reader = imageio.get_reader(video_path, "ffmpeg")

        video_name = video_file.split(".")[0]

        # os.makedirs(f"data/{video_name}/")

        for i, frame in enumerate(reader):
            pil_img = Image.fromarray(frame)
            pil_img.save(f"data/{video_name}/{video_name}{i}.jpg", format="JPEG")


gather()
# process("unprocessed_images")
