import os
import cv2
import torch
import asyncio
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from models.tag2text import ram
from inference_ram import inference as inference_ram
from inference_tag2text import tag2text_caption
from inference_tag2text import inference as inference_tag2text
from GRiT import setup_cfg
from detectron2.engine.defaults import DefaultPredictor


class FrameQualityEvaluator:
    def __init__(self, video, methods=None):
        self.video = video
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.total_seconds = int(self.total_frames / self.fps)

        # map method names to method implementations
        self.methods = {
            "motion_blur_detection": self.motion_blur_detection,
            "contrast_assessment": self.contrast_assessment,
            "sharpness_assessment": self.sharpness_assessment
        }

        if methods is None:
            self.selected_methods = self.methods
        else:
            self.selected_methods = {method: self.methods[method] for method in methods}

    def _check_time_interval(self, start_time, end_time):
        if start_time < 0 or end_time > self.total_seconds:
            raise ValueError("Time interval is out of video's duration.")
        if start_time > end_time:
            raise ValueError("Start time should be less than end time.")

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

    def reset_video(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def evaluate(self, start_time, end_time):
        self._check_time_interval(start_time, end_time)

        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)

        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        scores = {method: [] for method in self.selected_methods.keys()}  # Store scores of each method
        frames = []
        times = []
        for i in range(start_frame, end_frame):
            ret, frame = self.video.read()
            if not ret:
                break

            frames.append(frame)
            times.append(i / self.fps)
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
        best_time = None
        for i, frame in enumerate(frames):
            total_score = sum(scores[method_name][i] for method_name in scores.keys())
            if total_score > best_score:
                best_score = total_score
                best_frame = frame
                best_time = times[i]

        self.reset_video()  # reset video to the first frame after evaluation
        return best_frame, best_time


class VideoFrameExtractor:
    def __init__(self, video, methods=None):
        self.video = video
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.total_seconds = int(self.total_frames / self.fps)
        self.evaluator = FrameQualityEvaluator(video, methods=methods)

    def _check_time_interval(self, start_time, end_time):
        if start_time < 0 or end_time > self.total_seconds:
            raise ValueError("Time interval is out of video's duration.")
        if start_time > end_time:
            raise ValueError("Start time should be less than end time.")

    def get_best_frame_in_interval(self, start_time, end_time, save_path=None):
        self._check_time_interval(start_time, end_time)
        best_frame, best_time = self.evaluator.evaluate(start_time, end_time)
        if save_path:
            timestamp = int(best_time)
            filename = f"best_frame_{timestamp}_{timestamp + 1}.jpg"
            cv2.imwrite(os.path.join(save_path, filename), best_frame)
        return best_frame

    def extract_best_frames_per_second(self, output_folder=None):
        best_frames = []
        for sec in range(self.total_seconds):
            best_frame = self.get_best_frame_in_interval(sec, sec + 1, output_folder)
            best_frames.append(best_frame)
        return best_frames


class VideoDescriptionGenerator:
    def __init__(self,
                 ram_pretrained=r'E:\model_zoo\pretrained\recognize-anything\ram_swin_large_14m.pth',
                 tag2text_pretrained=r'E:\model_zoo\pretrained\recognize-anything\tag2text_swin_14m.pth',
                 thre=0.68,
                 image_size=384,
                 device='cuda'):
        self.image_size = image_size
        self.device = device

        # load ram model
        ram_model = ram(pretrained=ram_pretrained, image_size=image_size, vit='swin_l')
        ram_model.eval()
        self.ram_model = ram_model.to(device)

        # delete some tags that may disturb captioning
        # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
        delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]

        # load tag2text model
        tag2text_model = tag2text_caption(pretrained=tag2text_pretrained,
                                          image_size=image_size,
                                          vit='swin_b',
                                          delete_tag_index=delete_tag_index)
        tag2text_model.threshold = thre  # threshold for tagging
        tag2text_model.eval()
        self.tag2text_model = tag2text_model.to(device)

    def generate_video_description(self, extractor, predictor):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(), normalize
        ])

        total_seconds = int(extractor.total_seconds)
        descriptions = ''
        # descriptions = []
        english_tags = []
        chinese_tags = []
        for i in range(total_seconds):
            with torch.no_grad():
                best_frame = extractor.get_best_frame_in_interval(i, i + 1)

                predictions = predictor(best_frame)

                best_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
                best_frame = Image.fromarray(best_frame)

                best_frame = best_frame.resize((self.image_size, self.image_size))
                best_frame = transform(best_frame).unsqueeze(0).to(self.device)

                # 生成tags
                res_ram = inference_ram(best_frame, self.ram_model)
                # print("Image Tags: ", res_ram[0])
                # print("图像标签: ", res_ram[1])
                english_tags.append(res_ram[0])
                chinese_tags.append(res_ram[1])
                words = res_ram[0].split(" | ")
                tags = ','.join(words[:])  # min(5, len(words))
                # print("tags: ", tags)

                # 生成caption
                res_tag2text = inference_tag2text(best_frame, self.tag2text_model, tags)
                # print("Model Identified Tags: ", res_tag2text[0])
                # print("User Specified Tags: ", res_tag2text[1])
                # print("Image Caption: ", res_tag2text[2])
                caption = res_tag2text[2]

                # start_time = f"{i // 60:02d}:{i % 60:02d}"
                # end_time = f"{(i + 1) // 60:02d}:{(i + 1) % 60:02d}"
                # descriptions += f"{start_time}-{end_time}: {caption}.\n"
                descriptions += f"{caption}.\n"

                # start_time = f"{i:02d}:{(i * 60) % 60:02d}"
                # end_time = f"{(i + 1):02d}:{((i + 1) * 60) % 60:02d}"
                # descriptions += f"{start_time}-{end_time}: {caption}.\n"
                # descriptions += f"Second {i + 1}: {caption}.\n"

                for idx, bbox in enumerate(predictions['instances'].pred_boxes.tensor):
                    descriptions += f"{predictions['instances'].pred_object_descriptions.data[idx]}: {[int(i) for i in bbox.floor().tolist()]};"
                descriptions += f"\n"
            # descriptions.append(caption)
            del best_frame

        return english_tags, chinese_tags, descriptions


class AsyncVideoDescriptionGenerator:
    def __init__(self,
                 ram_pretrained=r'E:\model_zoo\pretrained\recognize-anything\ram_swin_large_14m.pth',
                 tag2text_pretrained=r'E:\model_zoo\pretrained\recognize-anything\tag2text_swin_14m.pth',
                 thre=0.68,
                 image_size=384,
                 device='cuda'):
        self.image_size = image_size
        self.device = device

        # load ram model
        ram_model = ram(pretrained=ram_pretrained, image_size=image_size, vit='swin_l')
        ram_model.eval()
        self.ram_model = ram_model.to(device)

        # delete some tags that may disturb captioning
        # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
        delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]

        # load tag2text model
        tag2text_model = tag2text_caption(pretrained=tag2text_pretrained,
                                          image_size=image_size,
                                          vit='swin_b',
                                          delete_tag_index=delete_tag_index)
        tag2text_model.threshold = thre  # threshold for tagging
        tag2text_model.eval()
        self.tag2text_model = tag2text_model.to(device)

    async def generate_video_description(self, extractor, predictor):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(), normalize
        ])

        descriptions = ''
        english_tags = []
        chinese_tags = []

        best_frame = extractor.get_best_frame_in_interval(0, 1)

        best_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
        best_frame = Image.fromarray(best_frame)

        best_frame = best_frame.resize((self.image_size, self.image_size))
        best_frame = transform(best_frame).unsqueeze(0).to(self.device)

        # 异步执行生成tags和caption
        async def generate_tags_caption():
            # 生成tags
            res_ram = await asyncio.to_thread(inference_ram, best_frame, self.ram_model)
            english_tags.append(res_ram[0])
            chinese_tags.append(res_ram[1])
            words = res_ram[0].split(" | ")
            tags = ','.join(words[:])

            # 生成caption
            res_tag2text = await asyncio.to_thread(inference_tag2text, best_frame, self.tag2text_model, tags)
            caption = res_tag2text[2]

            return caption

        # 并行执行生成tags和caption
        caption_task = asyncio.create_task(generate_tags_caption())

        # 同步执行预测
        predictions = await asyncio.to_thread(predictor, extractor.get_best_frame_in_interval(0, 1))

        # 等待生成caption的任务完成
        caption = await caption_task
        descriptions += f"{caption}.\n"

        for idx, bbox in enumerate(predictions['instances'].pred_boxes.tensor):
            descriptions += f"{predictions['instances'].pred_object_descriptions.data[idx]}: {[int(i) for i in bbox.floor().tolist()]};"
        descriptions += f"\n"

        return english_tags, chinese_tags, descriptions
        
        
def generate_video_description(video_path='./demo.mp4'):
    video = cv2.VideoCapture(video_path)
    # 视频转换为每秒一张图片，methods为选取指标，计算指标越少越快
    extractor = VideoFrameExtractor(video, methods=["motion_blur_detection", "contrast_assessment", "sharpness_assessment"])
    # 生成视频描述，tags用ram模型生成，caption用tag2text模型生成
    # tag2text模型可以直接生成tags，但是ram模型生成的tags更准确，因此用了两个模型
    vdg = VideoDescriptionGenerator(
        ram_pretrained=r'E:\model_zoo\pretrained\recognize-anything\ram_swin_large_14m.pth',
        tag2text_pretrained=r'E:\model_zoo\pretrained\recognize-anything\tag2text_swin_14m.pth',
        device='cuda'
    )
    # 初始化GRiT模型，生成目标描述
    cfg = setup_cfg()
    grit_predictor = DefaultPredictor(cfg)
    english_tags, chinese_tags, descriptions = vdg.generate_video_description(extractor, grit_predictor)
    return english_tags, chinese_tags, descriptions


def text_wrap(text, font, max_width):
    lines = []
    if font.getsize(text)[0] <= max_width:
        lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + " "
                i += 1
            if not line:
                line = words[i]
                i += 1
            lines.append(line)
    return lines


def add_text_to_video(video_path, output_path, english_tags, chinese_tags, descriptions):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # you may need to change this depending on your video codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    frame_counter = 0
    tag_index = 0
    color = (0, 0, 255)  # BGR, red
    font_path = "./STFangsong.ttf"  # replace with the path to a font file that supports Chinese characters

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to a PIL image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        english_tag = english_tags[min(tag_index, len(english_tags) - 1)]
        chinese_tag = chinese_tags[min(tag_index, len(chinese_tags) - 1)]
        description = descriptions[min(tag_index, len(descriptions) - 1)]

        # Define the area for each text and add the text to the image
        for i, text in enumerate([english_tag, chinese_tag, description]):
            y1 = (frame_size[1] // 3) * i
            y2 = (frame_size[1] // 3) * (i + 1)
            max_width = frame_size[0]
            font_size = 1
            font = ImageFont.truetype(font_path, font_size)
            while font.getsize(text)[0] < max_width:
                font_size += 1
                font = ImageFont.truetype(font_path, font_size)
            font_size -= 1
            font = ImageFont.truetype(font_path, font_size)
            lines = text_wrap(text, font, max_width)
            line_height = font.getsize('hg')[1]
            x = 0
            y = y1
            for line in lines:
                draw.text((x, y), line, font=font, fill=color)
                y += line_height

        # Convert the PIL image back to an OpenCV image
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(frame)

        # Move to the next tag every fps frames
        frame_counter += 1
        if frame_counter >= fps:
            frame_counter = 0
            tag_index += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    english_tags, chinese_tags, descriptions = generate_video_description(video_path='./demo.mp4')
    print("Video Description: ")
    print(descriptions)
    # add_text_to_video('H:/whole.mp4', './output.mp4', english_tags, chinese_tags, descriptions)

'''
在当前问题中，你将扮演一位人工智能专家，擅长的领域是自然语言处理、视频理解、多模态。
现在有一段视频，我将做如下操作：
1.视频每秒选质量最好的一帧，把这帧图片输入到一个多模态模型中，模型推理生成一句相关的文字描述。
2.现在把这些文字描述给你，视频的每秒都对应一条文字描述。
你要根据自己的经验，告诉我这段视频讲了什么。这其中有一些问题，你在思考的时候需要注意：
1.你需要考虑图片生成文字描述的局限性：从图片生成的文字描述丢失了视频中原有的运动信息，多模态模型的输入是图片不是视频，看不到一秒内物体是如何运动的。
2.你需要从视频连续性上思考：输入多模态模型的图片有明确时序特征，这些时序特征应当一定程度上指导了生成的文字描述。
3.你需要通过自己的分析去掉时序描述中不合人类逻辑的信息和图片生成文字描述中可能出现的误导信息。
4.我给你的文字描述是英文的。
5.你需要用中文尽最大可能还原出视频内容。
下面是视频全部的文字描述：
Second 1:a man playing baseball swings a bat at a ball while another person stands behind the net
Second 2:a man playing baseball swings a bat at a ball in a cage
Second 3:a man swings a baseball bat at a game while another player behind the net stands behind him and holds a tennis racket
Second 4:a man playing baseball swings a bat at a ball while a boy in a baseball glove stands behind a fence
Second 5:a man playing baseball swings a bat at a ball while a boy stands behind a fence
Second 6:a man playing baseball swings a bat at a ball while another person stands behind him
Second 7:a man playing baseball swings a bat at a ball while another person stands behind him
Second 8:a man playing baseball swings his bat while the batter stands behind him
Second 9:a man playing baseball swings a bat at a ball while another person stands behind him
Second 10:a man playing baseball swings his bat at a ball while the catcher catches the ball behind the cage

在当前问题中，你将扮演一位人工智能专家，擅长的领域是自然语言处理、视频理解、多模态。
现在有一段视频，我将做如下操作：
1.视频每秒选质量最好的一帧，把这帧图片输入到一个多模态模型中，模型推理生成一句和这张图片相关的文字描述。
2.这句描述下面，是图片中所有目标的“描述： 检测框坐标（左上角x、y，右下角x、y）“
3.现在把所有时间段的文字描述给你，视频的每秒都对应一条整体文字描述和很多目标的文字描述及坐标。
你要根据自己的经验，告诉我这段视频讲了什么。这其中有一些问题，你在思考的时候需要注意：
1.你需要考虑图片生成文字描述的局限性：从图片生成的文字描述丢失了视频中原有的运动信息，多模态模型的输入是图片不是视频，看不到一秒内物体是如何运动的。
2.你需要从视频连续性上思考：输入多模态模型的图片有明确时序特征，这些时序特征应当一定程度上指导了生成的文字描述。
3.你需要通过自己的分析去掉时序描述中不合人类逻辑的信息和图片生成文字描述中可能出现的误导信息。
4.我给你的文字描述是英文的。
5.你需要用中文尽最大可能还原出视频内容。
下面是视频全部的文字描述：
Second 1:
a man playing baseball swings a bat at a ball while another person stands behind the net.
man holding a red frisbee: [86, 53, 194, 242]
the shorts are black in color: [105, 151, 173, 202]
a white short sleeve shirt: [1, 92, 56, 274]
a man holding a ball: [271, 15, 424, 318]
white lines on the ground: [34, 179, 235, 283]
man wearing white sneakers: [87, 219, 185, 242]
a gray shirt on a man: [114, 81, 186, 161]
a game of tennis is taking place: [2, 2, 423, 318]
the blue shorts the man is wearing: [1, 267, 44, 319]
yellow tennis ball: [271, 161, 325, 227]
the socks are black: [91, 201, 177, 234]
green tennis court with white lines: [9, 172, 417, 318]
a black baseball cap: [125, 58, 154, 82]
a white baseball: [220, 270, 241, 291]
a red and white book: [159, 55, 182, 87]
a white frisbee: [97, 203, 149, 222]
woman holding a red and white object: [155, 52, 189, 107]
the roof of the building: [1, 0, 420, 35]
Second 2:
a man playing baseball swings a bat at a ball in a cage.
man holding a bat: [87, 30, 195, 243]
a mans plaid shorts: [108, 151, 174, 203]
the bat is wooden: [119, 25, 185, 101]
the white short sleeved shirt: [1, 90, 59, 275]
white lines on the ground: [37, 180, 235, 284]
man wearing white sneakers: [88, 219, 186, 243]
a grey shirt on a man: [116, 83, 183, 162]
the blue shorts the man is wearing: [1, 267, 45, 319]
green tennis court with white lines: [16, 172, 399, 319]
a white frisbee: [100, 203, 150, 222]
a game of tennis is taking place: [2, 2, 422, 318]
yellow tennis ball in the air: [274, 162, 328, 230]
a white baseball: [219, 271, 241, 292]
the socks are black: [95, 202, 175, 232]
yellow flowers: [263, 163, 351, 248]
a white roof: [2, 0, 417, 36]
a white sign on the wall: [159, 55, 185, 96]
a tall wire fence: [276, 3, 423, 317]
Second 3:
a man swings a baseball bat at a game while another player behind the net stands behind him and holds a tennis racket.
man is serving ball: [104, 4, 225, 199]
a man wearing blue shorts: [1, 0, 95, 320]
a pair of blue shorts: [2, 251, 84, 319]
black shorts on a man: [121, 108, 179, 162]
yellow bag on floor: [283, 122, 339, 192]
green field with white stripes: [70, 142, 243, 251]
a green soccer field: [72, 130, 415, 318]
man wearing a gray shirt: [115, 39, 192, 116]
people playing tennis: [3, 0, 421, 317]
black and white sneakers: [106, 160, 204, 195]
a white baseball: [244, 239, 264, 258]
a person in a yellow shirt: [362, 65, 394, 123]
the white towel on the mans head: [161, 12, 185, 43]
Second 4:
a man playing baseball swings a bat at a ball while a boy in a baseball glove stands behind a fence.
man in white tshirt with red design: [43, 1, 247, 320]
tennis player on court: [253, 46, 405, 241]
the man is wearing blue pants: [97, 265, 219, 319]
black shorts on the man: [278, 138, 351, 196]
a wooden tennis racket: [360, 55, 404, 134]
green floor of the tennis court: [43, 162, 422, 318]
a red and white sign on the fence: [0, 133, 46, 239]
the men are playing tennis: [8, 0, 419, 318]
a gray shirt on a man: [298, 76, 368, 152]
the head of a person: [125, 0, 199, 68]
the man is wearing black socks: [251, 193, 356, 246]
a tennis net: [7, 0, 423, 23]
white lines on the ground: [211, 175, 404, 290]
red number on blue shorts: [103, 270, 126, 320]
a gray baseball cap: [325, 46, 352, 72]
a chain link fence: [1, 102, 87, 319]
Second 5:
a man playing baseball swings a bat at a ball while a boy stands behind a fence.
man in white tshirt: [55, 14, 259, 320]
man wearing black shorts: [261, 61, 383, 259]
a red and white sign on the fence: [1, 138, 58, 255]
a pair of blue shorts: [110, 281, 230, 320]
black shorts on the tennis player: [286, 159, 357, 214]
the game being played is tennis: [8, 5, 420, 318]
green floor in the room: [52, 174, 422, 318]
a head of a person: [136, 10, 214, 90]
a gray shirt on a man: [302, 90, 376, 173]
a white tiled wall: [7, 0, 423, 38]
a long gray pole: [32, 98, 100, 317]
white lines on the ground: [223, 183, 420, 318]
a yellow arm on a chair: [1, 83, 66, 128]
a white and red soccer ball: [280, 213, 325, 234]
the man is wearing white shoes: [259, 208, 368, 263]
Second 6:
a man playing baseball swings a bat at a ball while another person stands behind him.
tennis player jumping in the air: [149, 42, 278, 232]
the short sleeved white shirt: [1, 75, 145, 299]
black shorts on a man: [164, 142, 223, 194]
a green and white tennis court: [3, 160, 422, 319]
a pair of blue pants: [4, 287, 128, 319]
two people playing tennis: [3, 3, 314, 317]
yellow chairs beside the court: [326, 158, 386, 226]
white lines on the ground: [114, 168, 285, 288]
the back of a mans head: [23, 4, 96, 74]
a mans black shirt: [163, 69, 244, 147]
the man is wearing sneakers: [148, 196, 252, 233]
blue tarp over metal fence: [2, 2, 423, 177]
a white and blue net: [1, 0, 417, 26]
a white baseball: [286, 276, 305, 295]
a black sock on a man: [221, 193, 243, 217]
two men sitting down: [367, 97, 424, 200]
man wearing black shorts: [153, 137, 245, 207]
Second 7:
a man playing baseball swings a bat at a ball while another person stands behind him.
tennis player serving ball: [158, 38, 271, 239]
a white short sleeved shirt: [2, 90, 151, 306]
black shorts on a man: [179, 150, 236, 201]
green flat field with white lines: [3, 168, 422, 318]
a pair of blue jeans: [11, 292, 138, 319]
white lines on the ground: [124, 178, 304, 295]
the back of a mans head: [25, 17, 98, 86]
a man playing tennis: [4, 7, 419, 317]
a metal fence: [2, 1, 415, 34]
a yellow tennis racket: [350, 159, 402, 230]
the man is wearing sneakers: [162, 205, 268, 240]
the small tan and red soccer ball: [167, 207, 218, 227]
Second 8:
a man playing baseball swings his bat while the batter stands behind him.
man in white shirt: [2, 0, 174, 319]
the man is holding a bat: [181, 37, 325, 230]
the mans blue shorts: [27, 262, 150, 319]
black shorts on the man: [203, 126, 277, 184]
green surface of tennis court: [3, 153, 416, 319]
a person is standing: [360, 128, 424, 319]
a mans short sleeve shirt: [221, 63, 285, 142]
the bat is made of wood: [267, 66, 323, 136]
a white fence: [16, 1, 398, 17]
blue tarp on a metal fence: [4, 2, 415, 161]
the back of a mans head: [47, 0, 121, 58]
the man is wearing white shoes: [181, 186, 282, 228]
the game being played is tennis: [5, 1, 358, 317]
white lines on the ground: [142, 165, 324, 272]
a small red and white baseball base: [191, 188, 239, 208]
a white baseball: [317, 263, 336, 284]
a bunch of yellow tennis balls: [362, 147, 420, 218]
a black baseball cap: [243, 37, 275, 65]
red design on blue shorts: [31, 267, 56, 319]
Second 9:
a man playing baseball swings a bat at a ball while another person stands behind him.
tennis player standing on the court: [181, 30, 293, 234]
man wearing white shirt: [2, 0, 134, 319]
the man is wearing black shorts: [203, 137, 278, 191]
a pair of blue shorts: [5, 270, 94, 319]
a white net: [49, 1, 392, 20]
the field is green: [102, 159, 407, 319]
gray shirt of tennis player: [218, 67, 287, 148]
man wearing white sneakers: [183, 211, 283, 234]
the game being played is tennis: [2, 0, 420, 319]
a yellow tennis racket: [365, 145, 421, 216]
white lines on the ground: [109, 163, 324, 274]
the man is wearing black socks: [183, 194, 283, 228]
a wooden baseball bat: [273, 23, 292, 95]
blue tennis court wall: [14, 0, 416, 158]
the emblem on the net: [188, 192, 241, 216]
a white baseball on the ground: [323, 265, 343, 285]
a man with black hair: [1, 0, 58, 69]
the head of a person: [232, 39, 263, 69]
Second 10:
a man playing baseball swings his bat at a ball while the catcher catches the ball behind the cage.
man holding a bat: [112, 37, 230, 260]
black shorts on the man: [138, 168, 202, 219]
white shirt on a man: [1, 99, 64, 289]
a yellow tennis racket: [292, 174, 348, 245]
two people sitting on a bench: [335, 114, 413, 213]
white lines on the ground: [50, 188, 256, 313]
a pair of blue shorts: [1, 276, 68, 320]
the field is green: [16, 186, 420, 318]
the net is black: [2, 3, 421, 316]
a black baseball cap: [157, 77, 190, 100]
grey shirt on man: [149, 99, 209, 179]
a wooden baseball bat: [163, 32, 225, 117]
a white net over a tennis court: [2, 0, 421, 48]
the man is wearing black socks: [115, 214, 201, 251]
the home plate of a baseball diamond: [121, 219, 167, 240]
blue tarp over a metal fence: [2, 36, 422, 190]
'''
