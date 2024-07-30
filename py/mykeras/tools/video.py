
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def make_video(frames, filename='output.mp4', fps=30):
#     height, width, channels = frames[0].shape
#     print(f"Making video {width}x{height}x{channels} {filename}")
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
#     for frame in frames:
#         # cv2.imshow('frame', frame)
#         to_write = cv2.calcHist([frame], [0], None, [256], [0, 256])
#         out.write(cv2.)
#         # out.write(cv2.getOptimalNewCameraMatrix(frame, (1., 1.), (height,width), 1.0, newImgSize=(height,width)))
#     out.release()

def create_video(frames, filename='output.mp4', fps=30):
    height, width, channels = frames[0].shape
    print(f"Creating video {width}x{height}x{channels} {filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    
def read_video(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


class VideoTesting(tf.test.TestCase):
    
    def test_make_video(self):
        frames = [
            np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8) for _ in range(1000)]
        print(frames[0].shape)
        # plt.plot(frames)
        # plt.imshow(frames[0])
        create_video(frames, fps=15)
        frames2 = read_video('output.mp4')
        # self.assertAllEqual(frames, frames2)
        # plt.plot(frames2[0][0])
        

    
if __name__ == '__main__':
    tf.test.main()
