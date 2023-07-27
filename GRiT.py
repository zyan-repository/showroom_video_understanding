import argparse
import multiprocessing as mp
import os
import cv2
import sys
import torch

from detectron2.config import get_cfg

# sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
current_file_path = os.path.abspath(__file__)
base_path = os.path.dirname(current_file_path)
centernet_module_path = os.path.join(base_path, 'third_party', 'CenterNet2', 'projects', 'CenterNet2')
sys.path.insert(0, centernet_module_path)
from centernet.config import add_centernet_config
from grit.config import add_grit_config

from grit.predictor import Visualizer_GRiT

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

# constants
WINDOW_NAME = "GRiT"


def setup_cfg():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(os.path.join(base_path, 'configs', 'GRiT_B_DenseCap.yaml'))
    # cfg.merge_from_list(['MODEL.WEIGHTS', 'model_zoo/grit_b_densecap.pth'])
    cfg.merge_from_list(['MODEL.WEIGHTS', os.path.join(base_path, 'model_zoo', 'grit_b_densecap.pth')])
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    img = cv2.imread(
        r'H:\workspace\GRiT\demo_images\time_20230308164835_weight_31.0_lenght_67.0_width_22.0_height_48.0_fps_000408_orbbec_rgb.jpg')
    predictions = predictor(img)

    img = img[:, :, ::-1]
    visualizer = Visualizer_GRiT(img, instance_mode=ColorMode.IMAGE)
    instances = predictions["instances"].to('cpu')
    vis_output = visualizer.draw_instance_predictions(predictions=instances)

    visualized_output = vis_output
    # bboxes
    print(predictions['instances'].pred_boxes.tensor)
    # descriptions
    print(predictions['instances'].pred_object_descriptions.data)

    # pred_object_descriptions
    output = r'H:\workspace\GRiT\visualization'
    if output:
        if not os.path.exists(output):
            os.makedirs(output)
        out_filename = os.path.join(output, 'test.jpg')
        visualized_output.save(out_filename)
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        cv2.waitKey(0)
