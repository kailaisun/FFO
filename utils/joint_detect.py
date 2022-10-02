import numpy as np
from scipy.optimize import linear_sum_assignment
def joint_de(head_info,other_info,thresh=0.8,conf=0.6,thresh1=0.8):
    info_head = {}
    head_boxes = head_info['boxes']
    body_boxes = other_info['filter_boxes']
    body_scores = other_info['filter_scores']
    other_boxes = other_info['other_boxes']
    head_scores = head_info['scores']

    # info_head = {}
    # head_boxes = other_info['filter_boxes']
    # body_boxes = other_info['filter_boxes']
    # body_scores = other_info['filter_scores']
    # other_boxes = other_info['other_boxes']
    # head_scores = other_info['filter_scores']
    if body_boxes.shape[0] > 0:

        iohs = matrix_ioh(head_boxes,body_boxes)
        ioh_max = np.amax(iohs, axis=1)
        joint_id = np.where(ioh_max>thresh)
        # joint_scores = head_scores[joint_id]
        # joint_boxes = head_boxes[joint_id]

        cost = iohs[joint_id]
        row_ind, col_ind = linear_sum_assignment(-cost)
        body_bool = np.array([True]*(body_boxes.shape[0]))
        body_bool[col_ind] = False
        unconfirm_body_boxes = body_boxes[body_bool]
        unconfirm_body_scores = body_scores[body_bool]
        body_true = unconfirm_body_scores>conf
        confirm_body_boxes = unconfirm_body_boxes[body_true]
        confirm_body_scores = unconfirm_body_scores[body_true]
        joint_boxes = head_boxes[row_ind]
        joint_scores = head_scores[row_ind]
        unconfirm_ids = np.where(ioh_max <= thresh)
        unconfirm_boxes = head_boxes[unconfirm_ids]
        unconfirm_scores = head_scores[unconfirm_ids]
    else:
        joint_boxes = np.empty((0,4))
        joint_scores = np.empty((0))
        confirm_body_boxes = np.empty((0,4))
        confirm_body_scores = np.empty((0))
        unconfirm_boxes = head_boxes
        unconfirm_scores = head_scores

    # unconfirm_ids = np.where(ioh_max<thresh)
    # unconfirm_boxes = head_boxes[unconfirm_ids]
    # unconfirm_scores = head_scores[unconfirm_ids]
    if other_boxes.shape[0]>0:

        iohs_false= matrix_ioh(unconfirm_boxes,other_boxes)
        ioh_false_max = np.amax(iohs_false, axis=1)
    else:
        ioh_false_max = np.zeros(len(unconfirm_boxes))
    confirm_id = np.where(ioh_false_max<=thresh1)
    miss_boxes = unconfirm_boxes[confirm_id]
    miss_scores = unconfirm_scores[confirm_id]

    # confirm_head_boxes =np.append(joint_boxes,confirm_body_boxes,axis=0)
    # confirm_head_scores = np.append(joint_scores,confirm_body_scores,axis=0)
    confirm_head_boxes =np.append(np.append(joint_boxes,miss_boxes,axis=0),confirm_body_boxes,axis=0)
    confirm_head_scores = np.append(np.append(joint_scores,miss_scores),confirm_body_scores,axis=0)

    info_head['boxes'] = confirm_head_boxes
    info_head['class_ids'] = np.zeros(len(confirm_head_boxes))
    info_head['raw_img'] = head_info['raw_img']
    info_head['img'] = head_info['img']
    info_head['scores'] = confirm_head_scores
    info_head['box_nums'] = len(confirm_head_boxes)
    if len(head_scores)!= len(confirm_head_scores):
        a = 1
    return info_head









def matrix_ioh(a, b):
    """
    return ioh of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + 1e-12)
