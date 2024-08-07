
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
    print(f"Creating video {filename} from frames of type {type(frames)}.")
    if True:
        height, width, channels = frames[0].shape
        print(f"Creating video {width}x{height}x{channels} {filename}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
    else:
        if isinstance(frames, pd.DataFrame):
            frames = frames.values
        elif isinstance(frames, list):
            frames = np.array(frames)
        elif isinstance(frames, np.ndarray):
            pass
        # elif isinstance(frames, dict):
        #     print(f"frames.keys() = {frames.keys()}")
        #     pass
        elif isinstance(frames, str):
            frames = pd.read_csv(frames).values
        elif isinstance(frames, tf.Tensor):
            frames = frames.numpy()
        else:
            raise ValueError(f"frames type {type(frames)} not supported")
        print(f"frames.shape = {frames.shape}")
        width, height, channels = frames[0].shape
        print(f"Creating video {width}x{height}x{channels} {filename}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, cv2.CAP_FFMPEG, fourcc, fps, (width, height))
        for frame in frames:
            print(f"Creating frame: frame.shape = {frame.shape}")
            # plt.imshow(frame)
            out.write(cv2.cvtColor(frame[:,:,:3], cv2.COLOR_RGB2BGR))
            # out.write(frame) # why doesn't this do anything? It's just an empty video.
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

def image_similarity(image1, image2):
    return tf.math.reduce_mean(tf.math.square(image1 - image2))


class VideoTesting(tf.test.TestCase):
    
    def test_make_video(self):
        frames = [
            np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        print(f"frames len = {len(frames)}")
        print(f"frames[0].shape = {frames[0].shape}")
        # plt.plot(frames)
        # plt.imshow(frames[0])
        create_video(frames, fps=10)
        frames2 = np.array(
            read_video('output.mp4')).reshape(-1, 480, 720, 3)
        print(f"frames2 len = {len(frames2)}")
        print(f"frames2[0].shape = {frames2[0].shape}")
        loss = tf.keras.losses.MeanSquaredError()(
            np.array(frames), frames2) / 480 / 720 / 3 / len(frames)
        print(f"loss per channel = {float(loss)}")
        self.assertLess(loss, 1e-3)
        # plt.plot(frames2[0][0])
        
    def test_image_similarity(self):
        image1 = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 255, (480, 720, 3), dtype=np.uint8)
        loss = image_similarity(image1, image2)
        print(f"loss = {loss}")
        self.assertLess(loss, 1e-3)

    
if __name__ == '__main__':
    tf.test.main()
