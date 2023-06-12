"""
Process module
Contains the video_to_array function which is used to process the video files
into numpy arrays for the model to predict on.
"""
from pathlib import Path

import cv2
import numpy as np

from app.modules.util.log import log


def video_to_array(
    video: Path,
    resize: tuple = (15, 75, 75, 3),
    step: int = 2,
    max_frames: int = 15,
) -> np.ndarray:
    """
    Process the video file into a numpy array for the model to predict on.
    Args:
        video (Path): Path to the video file
        resize (tuple, optional): Resize the video frames to this shape.
            Defaults to (15, 75, 75, 3).
        step (int, optional): Step size to skip frames. Defaults to 2.
        max_frames (int, optional): Maximum number of frames to process.
            Defaults to 15.
    Returns:
        np.ndarray: Processed video file as a numpy array
    Raises:
        Exception: Failed to process video file to an array
    """

    if video is None:
        return None
    # log
    # print('video_to_array: ', video)
    log.debug("video_to_array: %s", video.name)

    video = cv2.VideoCapture(str(video))

    frames_n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    failed = False
    frames = []
    try:
        # video.set(cv2.CAP_PROP_POS_FRAMES, start)
        new_packet = []
        i = 0
        for _ in range(frames_n // max_frames):
            while video.isOpened():
                # log.debug(f'video_to_array: i = {i}')
                ret, frame = video.read()
                if not ret:
                    break
                if (
                    i % step != 0
                ):  # Works for 30 fps videos to capture only 15 fps
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
    except Exception as exp:
        failed = exp
        log.error("video_to_array: %s", exp)
    finally:
        video.release()
        cv2.destroyAllWindows()

    if failed is not False:
        raise failed
    return np.array(frames)
