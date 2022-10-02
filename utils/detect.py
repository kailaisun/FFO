import sys


import torch
import numpy as np
import cv2

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES,Filter_CLASSES
from yolox.data.datasets import voc_classes
from yolox.exp.build import get_exp_by_name,get_exp_by_file
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import argparse


COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)




class Detector():
    """ 图片检测器 """
    def __init__(self,args,classes=COCO_CLASSES):
        super(Detector, self).__init__()
        #devic='cuda:{:}'.format(args.device)
        self.classes=classes
        self.ckpt = args.ckpt

        self.device = torch.device('cuda:{:}'.format(args.device))if torch.cuda.is_available() else torch.device('cpu')
        if args.exp_files==None:
            self.exp=get_exp(args.exp_files,args.model)
        else:
            self.exp = get_exp(args.exp_files,args.model)
        self.test_size = self.exp.test_size  # TODO: 改成图片自适应大小
        print(self.test_size)
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(self.ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        print(self.exp.nmsthre)
        if args.conf is not None:
            self.exp.test_conf = args.conf
        if  args.nms is not None:
            self.exp.nmsthre=args.nms




    def detect(self, raw_img, filter = False):
        info = {}
        img, ratio = preproc(raw_img, self.test_size)
        info['raw_img'] = raw_img
        info['img'] = img

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre  # TODO:用户可更改
            )[0]
        if outputs is not None:
            outputs = outputs.cpu().numpy()
        else:
            info['boxes']=[]
            info['scores']=[]
            info['class_ids'] =[]
            info['box_nums']=0
            info['visual'] = info['raw_img']
            return info

        info['boxes'] = outputs[:, 0:4]/ratio
        info['scores'] = outputs[:, 4] * outputs[:, 5]
        info['class_ids'] = outputs[:, 6]
        #rint(info['class_ids'])
        info['box_nums'] = outputs.shape[0]
        if filter:

            self.filter_class = filter
            info = self.filter(info)
        # 可视化绘图
        # if visual:
        #     #raw_img = info['raw_img'].copy()
        #     info['visual'] = vis(info['raw_img'].copy(), info['boxes'], info['scores'], info['class_ids'], self.exp.test_conf, self.classes)
        return info
    def filter(self,info):
        if info['box_nums']>0:
            class_ids = []
            scores = []
            boxes = []

            for box, class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                if self.filter_class and self.classes[int(class_id)] not in self.filter_class:
                    continue
                boxes.append(box)
                class_ids.append(class_id)
                scores.append(score)
            info['boxes'] = np.array(boxes)
            info['scores'] = np.array(scores)
            info['class_ids'] = np.array(class_ids)
            info['box_nums'] = len(boxes)
        return info



if __name__=='__main__':
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-p', "--path", type=str, default="/mnt/disk1/liup/data/test/1.avi", help="choose a video")
    parser.add_argument('-m', "--model", type=str, default='yolox-s', help="choose a model")
    parser.add_argument('-c','--ckpt',type=str,default='weights/yolox_s.pth',help="weight")
    parser.add_argument('-d', '--device', type=int, default='1', help='choose a gpu')
    parser.add_argument('-e', '--exp_files', type=str, default=None,
                        help="pls input your expriment description file")
    parser.add_argument('-f', '--conf', type=float, default=0.5,
                        help="test conf")
    args = parser.parse_args()

    detector = Detector(args)

    img = cv2.imread('assets/dog.jpg')
    info= detector.detect(img)
    cv2.imwrite('1.jpg',info['visual'])
    print(info['scores'])
