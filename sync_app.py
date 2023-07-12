import cv2
import numpy as np
from vidgear.gears import NetGear, CamGear
from collections import deque
import time
from generate_video_description import VideoFrameExtractor, VideoDescriptionGenerator, setup_cfg, DefaultPredictor
from sklearn.preprocessing import MinMaxScaler
import threading
import queue
import datetime


# 创建一个队列用于存储待处理的帧
frame_queue = queue.Queue(maxsize=300)

# 定义一个函数用于从摄像头获取帧并将其放入队列
def capture_frame(video_camera):
    while True:
        frame = video_camera.get_frame()
        frame_queue.put(frame)


class Reconnecting_CamGear:
    def __init__(self, cam_address, reset_attempts=50, reset_delay=5):
        self.cam_address = cam_address
        self.reset_attempts = reset_attempts
        self.reset_delay = reset_delay
        self.options = {"CAP_PROP_FRAME_WIDTH": 640, "CAP_PROP_FRAME_HEIGHT": 480, "CAP_PROP_FPS": 30}
        self.source = CamGear(source=self.cam_address, **self.options).start()
        self.running = True

    def read(self):
        if self.source is None:
            return None
        if self.running and self.reset_attempts > 0:
            frame = self.source.read()
            if frame is None:
                self.source.stop()
                self.reset_attempts -= 1
                print(
                    "Re-connection Attempt-{} occured at time:{}".format(
                        str(self.reset_attempts),
                        datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"),
                    )
                )
                time.sleep(self.reset_delay)
                self.source = CamGear(source=self.cam_address, **self.options).start()
                # return previous frame
                return self.frame
            else:
                self.frame = frame
                return frame
        else:
            return None

    def stop(self):
        self.running = False
        self.reset_attempts = 0
        self.frame = None
        if not self.source is None:
            self.source.stop()
            
            
class VideoCamera(object):
    def __init__(self, video_address):
        self.video_address = video_address
        options = {"CAP_PROP_FRAME_WIDTH": 1080, "CAP_PROP_FRAME_HEIGHT": 720, "CAP_PROP_FPS": 30}
        self.capture = CamGear(source=self.video_address, stream_mode=True, **options).start()
        self.q = deque(maxlen=300)

        # 获取视频的帧率
        # self.fps = self.capture.stream.get(cv2.CAP_PROP_FPS)
        self.fps = 30
        self.frame_duration = 1 / self.fps  # 计算每一帧应该持续的时间

    def __del__(self):
        self.capture.stop()

    def get_frame(self):
        try:
            frame = self.capture.read()
            if frame is None:
                return None
            return frame
            if len(self.q) == self.q.maxlen:
                self.q.popleft()  # 如果队列满了，我们移除最旧的帧
            self.q.append(frame)
            return self.q.popleft()  # 获取队列中的一帧
        except Exception as e:
            print(f"Error while capturing frame: {e}")
            return None

    def get_frames_for_seconds(self, seconds):
        frames = []
        for _ in range(int(self.fps * seconds)):
            frame = self.get_frame()
            frames.append(frame)
        return frames


class AppExtractor:
    def __init__(self, methods=None):
        self.frames = []
        self.methods = {
            "motion_blur_detection": self.motion_blur_detection,
            "contrast_assessment": self.contrast_assessment,
            "sharpness_assessment": self.sharpness_assessment
        }

        if methods is None:
            self.selected_methods = self.methods
        else:
            self.selected_methods = {method: self.methods[method] for method in methods}
        self.total_seconds = 1

    def motion_blur_detection(self, frame):
        """
        Evaluate motion blur.
        运动模糊检测，score越大越清晰
        """
        # Use some edge detection method like Sobel operator to detect edges.
        # If there are less edges, it means more blur.
        sobel = cv2.Sobel(frame, cv2.CV_64F, 1, 1, ksize=5)
        blur_score = sobel.var()
        return blur_score

    def contrast_assessment(self, frame):
        """
        Evaluate contrast.
        对比度评估，score越大对比度越高
        """
        # A simple method would be to calculate the variance of the grayscale image.
        # Higher variance indicates higher contrast.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast_score = gray.var()
        return contrast_score

    def sharpness_assessment(self, frame):
        """
        Evaluate sharpness.
        图像清晰度评估，score越大越清晰
        """
        # One way is to calculate the variance of the Laplacian of the image.
        # The lower the variance, the blurrier the image.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = laplacian.var()
        return sharpness_score

    def set_frames(self, frames):
        self.frames = frames

    def get_best_frame_in_interval(self, start, end):
        return self.frames[0]
        scores = {method: [] for method in self.selected_methods.keys()}  # Store scores of each method
        frames = []
        for frame in self.frames:
            frames.append(frame)
            # Evaluate frame quality
            for method_name, method in self.selected_methods.items():
                score = method(frame)
                scores[method_name].append(score)

        # Normalize scores
        scaler = MinMaxScaler()
        for method_name in scores.keys():
            scores[method_name] = scaler.fit_transform(np.array(scores[method_name]).reshape(-1, 1)).ravel()

        best_score = -1
        best_frame = None
        for i, frame in enumerate(frames):
            total_score = sum(scores[method_name][i] for method_name in scores.keys())
            if total_score > best_score:
                best_score = total_score
                best_frame = frame
        return best_frame


options = {"bidirectional_mode": True, "max_retries": 20}

server = NetGear(logging=False, port="13000", **options)
# camera = VideoCamera(video_address="rtsp://admin:nxin1234@10.100.22.23:554/main")
camera = Reconnecting_CamGear(
    cam_address="rtsp://admin:nxin1234@10.100.22.23:554/main",
    reset_attempts=20,
    reset_delay=5,
)
# extra_message = "This is some extra data."
extractor = AppExtractor(methods=["contrast_assessment"])

vdg = VideoDescriptionGenerator(
    ram_pretrained='/home/nxin/sata/model_zoo/recognize-anything/ram_swin_large_14m.pth',
    tag2text_pretrained='/home/nxin/sata/model_zoo/recognize-anything/tag2text_swin_14m.pth',
    device='cuda'
)
cfg = setup_cfg()
grit_predictor = DefaultPredictor(cfg)

description_update_interval = 2  # Update descriptions every 2 seconds
last_description_update_time = time.time() - description_update_interval
descriptions = 'None'
#capture_thread = threading.Thread(target=capture_frame, args=(camera,))
#capture_thread.start()


while True:
    # 检查队列是否为空
    #if not frame_queue.empty():
    try:
        # 从队列中获取帧
        #frame = camera.get_frame()
        frame = camera.read()
        if frame is None:
            continue
        # Update descriptions every 2 seconds
        #current_time = time.time()
        #if current_time - last_description_update_time >= description_update_interval:
        #    extractor.set_frames([frame])  # the model may expect a list of frames
        #    english_tags, chinese_tags, descriptions = vdg.generate_video_description(extractor, grit_predictor)
        #    last_description_update_time = time.time()
        server.send(frame, message=descriptions)
    except Exception as e:
        print(f"Error while sending frame: {e}")
