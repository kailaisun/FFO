from utils.visual import plt_sq
from utils.data_load import read_num, read_detect
import numpy as np
import matplotlib.pylab as plt
import os
from utils.evaluate import evaluate
plt.switch_backend('agg')
def obj_tra(a, b):
    c = []
    for i in range(len(a) - 1):
        if a[i] != a[i + 1]:
            c.append(i)
    c.append(len(a))
    for i in range(len(c) - 1):
        if c[i + 1] - c[i] > 100:
            tmp = np.argmax(np.bincount(b[c[i]+1:c[i] + 20]))

            tmp = tmp - a[c[i] + 1]
            a[c[i]:] += tmp
    return a
data_track = {}
data_detect = {}
gt ={}
pre = {}
from matplotlib.ticker import MaxNLocator
for i in range(3):
    dir_ = str(i+1)+'.txt'

    data_track[i+1] = read_num(os.path.join('result/joint_top', dir_))
    data_detect[i+1] = read_detect(os.path.join('result/total', dir_))
    gt[i+1] =read_num(os.path.join('gt', dir_))
    pre[i+1] = obj_tra(data_track[i+1].copy(),data_detect[i+1].copy())
    evaluate(pre[i+1],gt[i+1])
    print(sum(gt[i+1]/len(gt[i+1])))
    with open(('result/end/'+dir_),'w') as f:
        for j in range(len(pre[i+1])):
            f.write('frame: {},num: {}\n'.format(j+1,pre[i+1][j]))
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(gt[i+1],linewidth = 4, label =
         'Truth', color='r')
    plt.plot(data_track[i+1],linewidth = 2, label =
         'Motion estimation', color='g')
    plt.plot(pre[i+1],linewidth = 2, label =
         'Ours', color='k')
    title = 'Test_'+str(i+1)

    plt.xlabel('Frames per 5')
    plt.ylabel('Occupants')

    if i==0:
        plt.legend(loc='upper right')
    plt.title(title)
    plt.show()
    plt.savefig('result/end/'+title+'.png')
    # from svglib.svglib import svg2rlg
    # from reportlab.graphics import renderPDF
    # svg_file='result/end/'+title+'.svg'
    # pdf_file = 'result/end/'+title+'.pdf'
    # drawing = svg2rlg(svg_file)
    # renderPDF.drawToFile(drawing, pdf_file)
