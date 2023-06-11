import cv2
import numpy as np
import random
from logging import getLogger as log
from pathlib import Path


def video_to_array(
    video: Path,
    resize: tuple = (15, 75, 75, 3),
    num_pkts: int = 3,
    step: int = 2,
    max_frames: int = 15,
) -> np.ndarray:
    # raise NotImplementedError

    if video is None:
        return None
    # log
    # print('video_to_array: ', video)
    log().debug(f"video_to_array: {video.name}")

    video = cv2.VideoCapture(video.__str__())

    frames_n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    failed = False
    frames = []
    try:
        # video.set(cv2.CAP_PROP_POS_FRAMES, start)
        new_packet = []
        i = 0
        for _ in range(frames_n // max_frames):
            while video.isOpened():
                # log().debug(f'video_to_array: i = {i}')
                ret, frame = video.read()
                if not ret:
                    break
                if i % step != 0:  # Works for 30 fps videos to capture only 15 fps
                    i += 1
                    continue
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                frame = frame[:, :, [2, 1, 0]]  # RGB
                new_packet.append(frame)

                if len(new_packet) == max_frames:
                    stacked = np.array(new_packet) / 255.0
                    frames.append(stacked)
                    new_packet.pop(0)
                    break

                i += 1
        frames = np.stack(frames, axis=0)
    except Exception as e:
        failed = True
        log().error(f"video_to_array: {e}")
    finally:
        video.release()
        cv2.destroyAllWindows()

    if failed:
        raise Exception("Failed to process video file to an array")
    return np.array(frames)
