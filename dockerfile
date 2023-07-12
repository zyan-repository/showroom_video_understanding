FROM huggingface/transformers-pytorch-gpu:4.29.2

RUN pip install --no-cache-dir \
    numpy==1.21 \
    timm==0.4.12 \
    pycocoevalcap \
    torchvision \
    Pillow \
    scipy \
    flask \
    av \
    flask_cors \
    fairscale==0.4.4 \
    opencv-python \
    lvis \
    boto3 

RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/detectron2.git

WORKDIR /app

COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python3", "async_app.py"]
