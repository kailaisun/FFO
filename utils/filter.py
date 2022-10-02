from yolox.data.datasets import COCO_CLASSES,Filter_CLASSES
from utils.box_tool import  matrix_ioh
import numpy as np
def filter(info,classes=COCO_CLASSES,filter_class='person'):
    tr=0.0  #person conf
    tre=0.0 #scene conf
    if info['box_nums'] > 0:
        class_ids = []
        scores = []
        boxes = []
        other_scores=[]
        other_boxes=[]
        other_ids=[]


        for box, class_id, score in zip(info['boxes'], info['class_ids'], info['scores']):
            if filter_class and classes[int(class_id)] in filter_class:

                boxes.append(box)
                class_ids.append(class_id)
                scores.append(score)
            elif classes[int(class_id)] in COCO_CLASSES:
                #print(classes[int(class_id)])

                other_boxes.append(box)
                other_scores.append(score)
                other_ids.append(class_id)
        filter_boxes = np.array(boxes)
        filter_scores = np.array(scores)
        filter_boxes = filter_boxes[filter_scores>tr]
        filter_scores = filter_scores[filter_scores>tr]

        other_scores = np.array(other_scores)
        other_boxes = np.array(other_boxes)
        other_ids = np.array(other_ids)

        other_boxes = other_boxes[other_scores>tre]

        other_ids = other_ids[other_scores>tre]
        other_scores = other_scores[other_scores > tre]
        info['filter_boxes'] = filter_boxes
        info['filter_scores']= filter_scores
        info['other_boxes'] = other_boxes
        info['other_ids'] = other_ids
        info['boxes'] = filter_boxes
        info['scores'] = filter_scores
        info['class_ids'] =class_ids
        #rint(info['class_ids'])
        info['box_nums'] = len(class_ids)


    return info

def head_filter(info,thresh=0.5):
    boxes = info['boxes']
    a = np.zeros((boxes.shape[0],2))
    a[:,0] = boxes[:,2]-boxes[:,0]
    a[:,1] = boxes[:,3]-boxes[:,1]
    b1 = a[:,0]<100
    c1 = b1
    b = a[:,0]/a[:,1]
    c = (b<1.1) & (b>0.7)
    d = c&c1
    boxes = boxes[d]
    info['boxes'] = boxes
    info['scores'] = info['scores'][d]
    info['class_ids'] = info['class_ids'][d]
    info['box_nums'] = len(boxes)
    if len(boxes)>0:
        ioh = matrix_ioh(boxes,boxes)

        tmp = ioh - np.eye(ioh.shape[0])
        row,col = np.where(tmp>thresh)
        miss_id=row[row-col>0]
        if len(miss_id>0):
            head_bool = np.array([True] * (boxes.shape[0]))
            head_bool[miss_id]=False
            boxes = boxes[head_bool]
            info['boxes'] = boxes.reshape(-1,4)
            info['scores'] = info['scores'][head_bool]
            info['class_ids'] = info['class_ids'][head_bool]
            info['box_nums'] = len(boxes)

    return info





