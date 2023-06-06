import cv2
import numpy as np
import random

def video_to_array(video, resize:tuple=(15,75,75,3),num_pkts: int = 3, step: int = 2, max_frames:int=15) -> np.ndarray:
    raise NotImplementedError
    """
    This function should take a video, as in a loaded video object loaded by a form, and return a numpy array of shape (1,15,75,75,3)
    what it does now is take a video path, and return a numpy array of shape (1,15,75,75,3)
    TODO: Help pls :)
    """
    if video is None:
        return None

    video = cv2.VideoCapture(video)

    frames_n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    failed = False
    frames = []
    try:
        start = random.randint(1, frames_n-(15+num_pkts)*step)
        video.set(cv2.CAP_PROP_POS_FRAMES, start)
        new_packet = []
        i = 0
        for _ in range(num_pkts):
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                if i % step != 0: # Works for 30 fps videos to capture only 15 fps
                    i += 1
                    continue
                frame = cv2.resize(frame, resize, interpolation = cv2.INTER_AREA)
                frame = frame[:, :, [2, 1, 0]] # RGB
                new_packet.append(frame)

                if len(new_packet) == max_frames:
                    stacked = np.array(new_packet)/255.
                    frames.append(stacked)
                    new_packet.pop(0)
                    break

                i+=1
        frames = np.stack(frames, axis = 0)
    except:
        failed = True
    finally:
        video.release()
        cv2.destroyAllWindows()

    return np.array(frames)[np.newaxis,...] if not failed else None
