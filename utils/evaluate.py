from collections import deque
def evaluate(arr_a,arr_b):
    NMAE = sum(abs(arr_b-arr_a)/(arr_a+arr_b+1))/len(arr_a)
    score = sum(arr_a==arr_b)/len(arr_a)
    print('NMAE: {:.3}, score: {:.3}'.format(NMAE,score))
    return NMAE,score


def acc(data_pre, data_gt):
    gt_frame = data_gt.frame.copy()

    gt_num = list(data_gt.num)
    pre_frame = data_pre.frame
    pre_num = data_pre.num

    tp = 0

    in_t = 0
    for i in range(len(pre_frame)):

        frame = [i for i in [pre_frame[i], pre_frame[i] - 1, pre_frame[i] + 1] if i in gt_frame]

        if len(frame) > 0:
            in_t += 1
            index_ = gt_frame.index(frame[0])
            if gt_num[index_] == pre_num[i]:
                tp += 1
                del gt_num[index_]
                del gt_frame[index_]

    return tp, in_t,len(gt_frame)





