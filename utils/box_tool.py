import numpy as np
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

def center(bboxs):
    #print(len(bboxs))
    a = np.zeros((len(bboxs),2))
    a[:,0] = (bboxs[:,0]+bboxs[:,2])//2
    a[:, 1] = (bboxs[:, 1]+ bboxs[:, 3]) // 2
    return a