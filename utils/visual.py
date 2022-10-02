from yolox.utils.visualize import vis, vis_head
import numpy as np
import cv2
import matplotlib.pylab as plt

def visual(img,info,cls_names,conf=0.2,color=None):
    img = img.copy()
    bboxes = info['boxes']
    scores = info['scores']
    cls = info['class_ids']
    if cls_names[0]=='head':
        vis_res = vis_head(img, bboxes, scores, cls, conf, cls_names, color=color)
    else:
        vis_res = vis(img, bboxes, scores, cls, conf,cls_names,)
    return vis_res

# def vis(img, boxes):
#     for i in range(len(boxes)):
#         box = boxes[i]
#
#         x0 = int(box[0])
#         y0 = int(box[1])
#         x1 = int(box[2])
#         y1 = int(box[3])
#
#         color = (0, 255, 0)
#
#         cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
#
#     return img


def midpiont(box):
    return (box[0] + box[2]) // 2, box[3]


def vis_track(img, boxes):
    for i in range(len(boxes)):
        box = boxes[i]

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        id = box[4]
        color_ = _COLORS[id % _COLORS.shape[0]]
        color = (color_ * 255).astype(np.uint8).tolist()
        text = '%d' % (id)
        txt_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (color_ * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.7, txt_color, thickness=2)
        mid = midpiont(box)
        #if mid[1] < 1080:
        cv2.circle(img, center=(mid[0], mid[1]), radius=4, color=(0, 255, 0), thickness=-1)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

def plt_sq(data,title=None):
    from matplotlib.ticker import MaxNLocator
    color = ['#FF0000','#00FF00','#0000FF','#000000']
    wid = ['4','2','1','2']
    i = 0
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    for k in data.keys():
        sq_num = data[k].sq_num
        name = data[k].name


        plt.plot(sq_num,linewidth = wid[i], label = name, color=color[i])
        i+=1
    #plt.rcParams['figure.figsize'] = (8.0, 6.0)
    plt.xlabel('Frames per 5')
    plt.ylabel('Occupants')
    if title=='Test_1':
        plt.legend(loc='upper right')
    if title:
        plt.title(title)
        plt.savefig('result/joint_top/'+title+'.png')
    plt.show()
    # from svglib.svglib import svg2rlg
    # from reportlab.graphics import renderPDF
    # svg_file='result/joint_top/'+title+'.svg'
    # pdf_file = 'result/joint_top/'+title+'.pdf'
    # drawing = svg2rlg(svg_file)
    # renderPDF.drawToFile(drawing, pdf_file)

