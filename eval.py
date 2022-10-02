import re,os
import numpy as np
from utils.evaluate import evaluate
from utils.data_load import read_num,read_detect
from tabulate import tabulate
def eval(pre_dir,gt_dir):
    pre_num = read_detect(pre_dir)
    gt_num = read_num(gt_dir)
    #gt_num [456:839]=4
    #fp_num = min(len(pre_num),len(gt_num))
    NMAE,score = evaluate(pre_num,gt_num)

    print('视频: ', pre_dir.split('/')[-1].split('.')[0],'NMAE: ',round(NMAE,3), 'score: ',round(score,3))
    return round(NMAE,3),round(score,3)

a = 1
# def read_num(dirs):
#     num = []
#
#     with open(dirs,'r') as f:
#         lines = f.readlines()
#     for i in range(len(lines)):
#         line = re.findall(r"\d+\.?\d*", lines[i])
#         num.append(int(line[1]))
#     num = np.array(num)
#     return num

# lab_dir = '2.txt'
# a = os.path.join('result/3_head',lab_dir)
# b = os.path.join('gt',lab_dir)
# eval(a,b)
# a = os.path.join('result/3_0.8',lab_dir)
# b = os.path.join('gt',lab_dir)
# eval(a,b)
#print('ioh=0.5')
# pre='result/toal/3.txt'
# gt= 'gt/3.txt'
if __name__=='__main__':
    # for i in range(3):
    #     pre = 'result/total1/'+str(i+1)+'.txt'
    #     gt = 'gt/'+str(i+1)+'.txt'
    #     eval(pre,gt)
    pre = 'result/toal1/'+'2.txt'
    gt = 'gt/'+'2.txt'
    eval(pre,gt)