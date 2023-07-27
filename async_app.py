from flask import Flask, render_template, Response, jsonify
import av
import os
import re
import cv2
import json
import platform
import subprocess
import numpy as np
import threading
import time
import queue
import torch
import logging
import asyncio
import concurrent.futures
from flask_cors import CORS
from logging.handlers import RotatingFileHandler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from generate_video_description import VideoFrameExtractor, VideoDescriptionGenerator, setup_cfg, DefaultPredictor


app = Flask(__name__)
CORS(app)


os.makedirs('logs/', exist_ok=True)
file_handler = RotatingFileHandler('logs/app.log', maxBytes=1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR)
# 将处理器添加到app的日志记录器中
app.logger.addHandler(file_handler)
with open('configs/configs.json') as json_file:
    configs = json.load(json_file)
os.environ["CUDA_VISIBLE_DEVICES"] = configs.get('CUDA_VISIBLE_DEVICES', "0")
max_width, max_height = 1024, 1024
frame_queues = {}
camera_names_global = []
camera_uri_global = {}
latest_inference_result = {}
latest_category_label = {}
is_inference_thread_alive = False


def init_cameras():
    global camera_names_global
    global camera_uri_global
    camera_uris = configs.get('CAMERA_URIS', None).split(',')
    camera_names = []
    for camera_uri in camera_uris:
        camera_name, uri = camera_uri.split('=')
        camera_uri_global[camera_name] = uri
        frame_queues[camera_name] = queue.Queue(maxsize=1)
        camera_names.append(camera_name)
    camera_names_global = camera_names


class AppExtractor:
    def __init__(self, methods=None):
        self.frames = []
        self.total_seconds = 1  # always 1

    def set_frames(self, frames):
        self.frames = frames

    def get_best_frame_in_interval(self, start, end):
        return self.frames[0]
        

vdg = None
grit_predictor = None
device = None
tokenizer = None
translate_model = None
extractor = AppExtractor()


def load_models():
    global vdg
    global grit_predictor
    global device
    global tokenizer
    global translate_model
    
    vdg = VideoDescriptionGenerator(
        ram_pretrained='model_zoo/ram_swin_large_14m.pth',
        tag2text_pretrained='model_zoo/tag2text_swin_14m.pth',
        device='cuda'
    )
    cfg = setup_cfg()
    grit_predictor = DefaultPredictor(cfg)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained("model_zoo/opus-mt-en-zh")
    translate_model = AutoModelForSeq2SeqLM.from_pretrained("model_zoo/opus-mt-en-zh").to(device)
        
        
def translate_single_sentence(inp_text, tokenizer, model, device):
    batch = tokenizer.prepare_seq2seq_batch(src_texts=inp_text, return_tensors='pt', max_length=512).to(device)
    translation = model.generate(**batch)
    result = tokenizer.batch_decode(translation, skip_special_tokens=True)
    return result


def remove_duplicate_sentences(elements):
    seen_sentences = set()
    unique_elements = []
    # 根据分隔符";"将短句拆分为多个子句
    parts = elements.split(";")
    for part in parts:
        # 去除首尾的空格并添加到已见短句集合中
        stripped_part = part.strip()
        if stripped_part == '':
            continue
        if stripped_part[0].islower():
            stripped_part = stripped_part.capitalize()
        if stripped_part not in seen_sentences:
            unique_elements.append(stripped_part)
            seen_sentences.add(stripped_part)
    return unique_elements
    

def generate_category_label(caption, tags, elements):
    keywords = {
        '打电话': {
            'v': ['talk'],
            'n': ['phone', 'smartphone']
        },
        '人员摔倒': {
            'v': ['lay', 'stretch'],
            'n': ['ground', 'floor', 'carpet']
        },
        '检查猪': {
            'v': ['squat'],
            'n': ['pig', "piggy bank", 'pet']
        },
        '搬运猪': {
            'v1': ['catch', 'carry'],
            'v2': ['stand', 'walk']
        },
        '扫地': {
            'v': ['sweep', 'clean'],
            'n': ['broom', 'swab', 'shovel', 'vacuum', "golf club"]
        },
        '喝水': {
            'v': ['drink'],
            'n': ['beverage', 'alcohol', 'water', 'beer', 'soda', 'wine', 'bottle', 'can']
        },
    }
    weak_text = []
    weak_text.extend(re.split('\W+', caption))
    weak_text.extend(re.split('\W+', tags))
    weak_text.extend(re.split('\W+', elements))
    strong_text = []
    strong_text.extend(re.split('\W+', caption))
    strong_text.extend(re.split('\W+', tags))
    # 对 8 种事件进行判断，动词+名词模型
    if sum(key in weak_text for key in keywords['打电话']['v']) >= 1 and sum(key in weak_text for key in keywords['打电话']['n']) >= 1:
        return 2
    elif sum(key in weak_text for key in keywords['人员摔倒']['v']) >= 1 and sum(key in weak_text for key in keywords['人员摔倒']['n']) >= 1:
        return 3
    # 检查猪 squat petting  pig
    elif sum(key in strong_text for key in keywords['检查猪']['v']) >= 1 and sum(key in strong_text for key in keywords['检查猪']['n']) >= 1:
        return 4
    # 搬运猪 stand catch pig
    elif sum(key in strong_text for key in keywords['搬运猪']['v1']) >= 1 and sum(key in strong_text for key in keywords['搬运猪']['v2']) >= 1 and sum(key in strong_text for key in keywords['检查猪']['n']) >= 1:
        return 5
    # 观察猪
    elif sum(key in strong_text for key in keywords['检查猪']['n']) >= 1:
        return 6
    # 扫地
    elif sum(key in strong_text for key in keywords['扫地']['v']) >= 1 and sum(key in weak_text for key in keywords['扫地']['n']) >= 1:
        return 7
    # 喝水
    elif sum(key in strong_text for key in keywords['喝水']['v']) >= 1 and sum(key in weak_text for key in keywords['喝水']['n']) >= 1:
        return 8
    # 使用灭火器
    elif 'extinguisher' in weak_text:
        return 9
    
    return 1

    
def infer(camera_name, frame):
    global latest_inference_result
    global latest_category_label
    global is_inference_thread_alive
    try:
        # Your inference code here
        extractor.set_frames([frame])  # the model may expect a list of frames
        english_tags, chinese_tags, descriptions = vdg.generate_video_description(extractor, grit_predictor)
        descriptions = re.sub(r': \[\d+, \d+, \d+, \d+\]', '', descriptions)
        if descriptions[0].islower():
            descriptions = descriptions.capitalize()
        parts = descriptions.split(".", 1)
        # caption_out, elements_out = translate_single_sentence([parts[0].strip() + '.', remove_duplicate_sentences(parts[1].strip())], tokenizer, translate_model, device)
        caption_out = translate_single_sentence([parts[0].strip() + '.'], tokenizer, translate_model, device)[0]
        all_elements = remove_duplicate_sentences(parts[1].strip())
        elements = translate_single_sentence(all_elements, tokenizer, translate_model, device)
        elements_out = '\n'.join([f"{i}. {el1} {el2}" for i, (el1, el2) in enumerate(zip(all_elements, elements), start=1)])
    
        caption = "主题：\n" + parts[0].strip() + "." + "\n" + caption_out + "\n"
        tags = "标签：\n" + "".join(english_tags) + "\n" + "".join(chinese_tags) + "\n"
        elements = "要素：\n" + elements_out
        # return descriptions
        result = caption + tags + elements
        latest_category_label[camera_name] = generate_category_label(caption, tags, elements)
        latest_inference_result[camera_name] = result
        is_inference_thread_alive = True
    except Exception as e:
        is_inference_thread_alive = False
        app.logger.error(f"Inference error: {e}")
    finally:
        torch.cuda.empty_cache()
    
    
@app.route('/')
def index():
    global camera_names_global
    port = configs.get('PORT', "12005")
    return render_template('index.html', port=port, camera_names=camera_names_global)


def open_camera(uri):
    try:
        container = av.open(uri)
        return container
    except av.AVError as e:
        app.logger.error(f'Error opening source: {e}')
        return None
            

max_workers = int(configs.get('MAX_WORKERS', '1'))
infer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)


async def async_inference_worker(camera_name, executor):
    loop = asyncio.get_event_loop()
    while True:
        try:
            if not frame_queues[camera_name].empty():
                img = frame_queues[camera_name].get()
                await loop.run_in_executor(executor, infer, camera_name, img)
        except Exception as e:
            app.logger.error(f"Error in inference thread: {e}. Restarting...")
            await asyncio.sleep(1)


def start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def start_async_inference_workers():
    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=start_event_loop, args=(new_loop,))
    thread.start()
    global camera_names_global
    global infer_executor
    for camera_name in camera_names_global:
        asyncio.run_coroutine_threadsafe(async_inference_worker(camera_name, infer_executor), new_loop)          


@app.route('/video/<camera_name>')
def video(camera_name):
    def generate(camera_name):
        global camera_uri_global
        uri = camera_uri_global.get(camera_name)
        if not uri:
            return jsonify({"error": "Camera name not found"}), 404
        container = open_camera(uri)

        if container is None:
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')

        # Check the environment variable
        valid_area = configs.get('VALID_AREA', None)
        if valid_area is not None:
            valid_area = valid_area.split(',')
            valid_area = [int(coordinate) for coordinate in valid_area]
            valid_area = np.array(valid_area).reshape(-1, 2)
            
        stream = next(s for s in container.streams if s.type == 'video')
        is_live_stream = stream.duration is None

        fps = stream.average_rate
        frame_delay = 1.0 / fps if fps else 0

        for packet in container.demux():
            for frame in packet.decode():
                if frame is not None:
                    original_height, original_width = frame.to_ndarray(format='bgr24').shape[:2]
                    aspect_ratio = original_width / original_height

                    if (max_width / aspect_ratio) <= max_height:
                        new_width = max_width
                        new_height = int(new_width / aspect_ratio)
                    else:
                        new_height = max_height
                        new_width = int(new_height * aspect_ratio)

                    break
            if 'new_width' in locals():
                break

        while True:
            try:
                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format='bgr24')
                    
                    # Scale the valid area polygon
                    if valid_area is not None:
                        scale_x = new_width / original_width
                        scale_y = new_height / original_height
                        valid_area_scaled = valid_area.copy()
                        valid_area_scaled[:, 0] = valid_area[:, 0] * scale_x
                        valid_area_scaled[:, 1] = valid_area[:, 1] * scale_y
                        
                    img = cv2.resize(img, (new_width, new_height))
                    
                    # Draw the valid area polygon
                    if valid_area is not None:
                        img_poly = cv2.polylines(img.copy(), [valid_area_scaled], True, (0,0,255), 2)
                        mask = np.zeros_like(img)
                        cv2.fillPoly(mask, [valid_area_scaled], (255,255,255))
                        img_infer = cv2.bitwise_and(img, mask)
                        x_min, y_min = valid_area_scaled.min(axis=0)
                        x_max, y_max = valid_area_scaled.max(axis=0)
                        img_infer = img_infer[int(y_min):int(y_max), int(x_min):int(x_max)]
                        if frame.key_frame:
                            # Put into queue
                            try:
                                frame_queues[camera_name].put_nowait(img_infer)
                            except queue.Full:
                                frame_queues[camera_name].get()
                                frame_queues[camera_name].put_nowait(img_infer)
                    else:
                        if frame.key_frame:
                            # Put into queue
                            try:
                                frame_queues[camera_name].put_nowait(img)
                            except queue.Full:
                                frame_queues[camera_name].get()
                                frame_queues[camera_name].put_nowait(img)

                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    # result, encimg = cv2.imencode('.jpg', img_infer, encode_param)
                    result, encimg = cv2.imencode('.jpg', img_poly if valid_area is not None else img, encode_param)
                    if result:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + encimg.tobytes() + b'\r\n')

                    if not is_live_stream: # 只有视频文件才休眠
                        time.sleep(frame_delay)

                if not is_live_stream:
                    container.seek(0)
            except av.AVError as e:
                app.logger.error(f'Error reading from source: {e}. Reopening source...')
                container = open_camera(uri)
            finally:
                global is_inference_thread_alive
                if not is_inference_thread_alive:
                    app.logger.warning("Inference thread stopped. Restarting...")
                    time.sleep(1)
                    start_async_inference_workers()
                    
    return Response(generate(camera_name), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/inference/<camera_name>')
def inference(camera_name):
    global camera_uri_global
    global latest_inference_result
    global latest_category_label

    uri = camera_uri_global.get(camera_name)
    if not uri:
        return jsonify({"error": "Camera name not found"}), 404  

    result = latest_inference_result.get(camera_name, "无推理结果")
    category = latest_category_label.get(camera_name, 1)

    return jsonify(result=result, category=category)


if __name__ == "__main__":
    init_cameras()
    load_models()
    start_async_inference_workers()
    port = int(configs.get('PORT', "12005"))
    app.run(host='0.0.0.0', port=port)
