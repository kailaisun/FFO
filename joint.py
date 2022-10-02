import re,os
import numpy as np
from eval import eval
from utils.evaluate import evaluate
from utils.joint_track import joint_tr
from utils.visual import plt_sq
from utils.data_load import load_data,filter,read_area_num,read_num,diff
import matplotlib.pyplot as plt
from datetime import datetime
plt.switch_backend('agg')
class data_sq:
    def __init__(self):
        self.sq_num = None
        self.name = None
def main():
    tim=datetime.now()
    print(tim)
    data = load_data()
    data_track = filter(data[0])
    data_top = data[1]
    area_num = []
    #  door area
    area_1 = [(1059, 222), (1410, 591)]

    area_3 = [(1600, 145), (1837, 511)]

    area = [area_1,area_1,area_3]
    for i in range(3):
        with open('result/total/'+str(i+1)+'.txt', 'r') as f:
            lines = f.read()
        f.close()
        a = lines.split('frame')
        a = a[1:]
        area_num.append(read_area_num(a, area[i]))
    stat = [0, 3, 3, 11]
    data_pre = {}
    frame_num = [796, 917, 837]
    for i in data_track.keys():
        log = []
        dir_ = str(i) + '.txt'
        # a = data_top[i].frame
        # b = data_top[i].num
        a, b = joint_tr(data_track[i], data_top[i],area_num[i-1])
        # print(data_track[i].frame)
        # print(data_top[i].frame)
        data_pre[i] = diff()
        data_pre[i].frame = a
        data_pre[i].num = b
        num = frame_num[i - 1]
        # print(a)
        # print(b)

        lab = np.zeros(num)

        sta = stat[i]
        lab[0:a[0]] = sta
        for i in range(len(a) - 1):
            sta += b[i]
            lab[a[i] - 1:a[i + 1]] = sta
        lab[a[i + 1]:num] = sta + b[i + 1]
        for i in range(len((lab))):
            text = 'frame: {}, num: {}'.format(i, int(lab[i]))
            log.append(text + '\n')
        with open(('result/joint_top/' + dir_), 'w+') as f:
            f.writelines(log)
    tim1 = datetime.now()
    print(tim1)
    print((tim1-tim).seconds)
    ccc=0
    ccc=(tim1-tim).seconds
    print(type((tim1-tim).seconds))
    eval('result/joint_top/1.txt', 'gt/1.txt')
    eval('result/track_top/1.txt', 'gt/1.txt')
    eval('result/joint_top/2.txt', 'gt/2.txt')
    eval('result/track_top/2.txt', 'gt/2.txt')
    eval('result/joint_top/3.txt', 'gt/3.txt')
    eval('result/track_top/3.txt', 'gt/3.txt')
    return data_pre,data_top,data_track
if __name__=='__main__':
    data = main()
    for i in range(3):
        sq_data = {}
        dir_ = str(i+1)+'.txt'
        sq_data[1] = data_sq()
        sq_data[1].sq_num = read_num(os.path.join('gt', dir_))
        sq_data[1].name = 'Truth'
        sq_data[2] = data_sq()
        sq_data[2].sq_num = read_num(os.path.join('result/track_top', dir_))
        sq_data[2].name = 'Overhead estimation'
        sq_data[3] = data_sq()
        sq_data[3].sq_num = read_num(os.path.join('result/deepsort', dir_))
        sq_data[3].name = 'Indoor estimation'
        sq_data[4] = data_sq()
        sq_data[4].sq_num = read_num(os.path.join('result/joint_top', dir_))
        sq_data[4].name = 'Ours'
        plt_sq(sq_data, title='Test_'+str(i+1))


