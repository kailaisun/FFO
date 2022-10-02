import re
import numpy as np

from utils.evaluate import evaluate
import matplotlib.pylab as plt

class diff:
    def __init__(self):
        self.frame = []
        self.num = []
        #self.area_num = []

    def append(self, content):
        if int(content[1])!=0:


            self.frame.append(int(content[0]))
            self.num.append(int(content[1]))
        # if len(content) > 2:
        #     self.area_num.append(int(content[2]))
        # else:
        #     self.area_num.append(0)


def read_txt(dir_):
    diff_ = diff()
    with open(dir_, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = re.findall(r"-?\d+\.?\d*", line)
            diff_.append(content)
    return diff_


def load_data():
    data_track = {}
    data_top = {}
    for i in range(3):
        dir_ = 'result/track/' + str(i + 1) + '_.txt'
        dir_1 = 'track_top/' + str(i + 1) + '.txt'
        data_track[i + 1] = read_txt(dir_)
        data_top[i + 1] = read_txt(dir_1)
    return data_track, data_top


def filter(data):
    for key in data.keys():
        frame = []
        num = []
        tmp = -1
        #area_num = []
        for i in range(len(data[key].frame)):
            frame_num = (data[key].frame[i]-1) // 5 + 1
            if frame_num !=tmp:
                frame.append(frame_num)
                num.append(data[key].num[i])
                #area_num.append(data[key].area_num[i])

            else:
                num[-1] += data[key].num[i]
                #area_num[-1] = max(area_num[-1],data[key].area_num[i])
            tmp = frame_num

        data[key].frame = frame
        data[key].num = num
    return data

def read_num(dirs):
    num = []

    with open(dirs,'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = re.findall(r"\d+\.?\d*", lines[i])
        if len(line)>1:
            num.append(int(line[1]))
        else:
            num.append(0)
    num = np.array(num)
    return num

def read_detect(dirs):
    num = []

    with open(dirs,'r') as f:
        lines = f.read()
    f.close()
    lines = lines.split('frame')
    lines = lines[1:]
    for i in range(len(lines)):
        line = re.findall(r"\d+\.?\d*",lines[i])
        num.append(int(line[1]))
    num = np.array(num)
    return num




def read_area_num(a,area):
    bboxes = []
    for line in a:
        bbox_line = line.split('bboxes')[-1]
        num = re.findall(r"\d+\.?\d*", bbox_line)
        bbox = np.zeros((len(num) // 4, 4))
        for i in range(len(num) // 4):
            num = list(map(float, num))
            bbox[i] = num[4 * i:4 * i + 4]
        bboxes.append(bbox)
    from utils.box_tool import center
    from utils.count import ifin

    area_arr = []
    for bbox in bboxes:
        area_num = 0
        if len(bbox) > 0:
            conter_point = center(bbox)

            for point in conter_point:
                if ifin(point, area[0], area[1]):
                    area_num += 1
        area_arr.append(area_num)
    return area_arr

def plt_sq(data,title=None):
    color = ['#FF0000','#00FF00','#0000FF','#000000']
    wid = ['4','2','1','2']
    i = 0

    for k in data.keys():
        sq_num = data[k].sq_num
        name = data[k].name

        plt.plot(sq_num,linewidth = wid[i], label = name, color=color[i])
        i+=1
    plt.rcParams['figure.figsize'] = (8.0, 6.0)
    plt.xlabel('Frames per 5')
    plt.ylabel('Occupants')
    plt.legend(loc='upper left')
    if title:
        plt.title(title)
        plt.savefig(title+'.svg')
    plt.show()