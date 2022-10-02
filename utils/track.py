import sys

from yolox.data.datasets.coco_classes import COCO_CLASSES
from utils.detect import Detector
from deep_sort.utils.parser import get_config
from yolox.data.datasets import COCO_CLASSES
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import numpy as np
import os
import  argparse


class Tracker():
    def __init__(self, args):

        self.ckpt=args.ckpt
        self.model=args.model
        self.device=args.device
        self.detector = Detector(args)
        cfg = get_config()
        cfg.merge_from_file("deep_sort/config/deep_sort.yaml")
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

    def track(self,image):
        info = self.detector.detect(image, filter='person')
        outputs = []
        if info['box_nums']>0:
            bbox_xywh = []
            scores = []
            #bbox_xywh = torch.zeros((info['box_nums'], 4))
            for (x1, y1, x2, y2), class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                # if self.filter_class and class_names[int(class_id)] not in self.filter_class:
                #     continue
                bbox_xywh.append([int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1])
                scores.append(score)
            bbox_xywh = torch.Tensor(bbox_xywh)
            if len(bbox_xywh):
                outputs = self.deepsort.update(bbox_xywh, scores, image)
        info['track'] = outputs
        return info








