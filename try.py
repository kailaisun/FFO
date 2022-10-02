import argparse
import cv2,os
import numpy as np
from detect import Detector
from yolox.data.datasets import COCO_CLASSES,Train_CLASSES
from yolox.exp import get_exp

# path = 'labels/2.txt'
# with open(path,'r') as f:
#     a = f.read()
# a = a.replace('\n','')
# b = a.replace(']',']\n')
# with open('labels/2.txt','w+') as f:
#     f.writelines(b)

lab11 = [1,36,37,44,54,77,166,189,214,225,658,678,689,770,775]
lab12 = [3,4,5,4,5,4,5,4,3,4,3,4,3,4,3]
lab21 = [1,11,14,26,27,28,32,45,50,60,113,148,158,164,170,171,173,213,255,256,259,262,278,281,283,372,845,857,897,898,899]
lab22 = [3,2,3,2,1,0,1,2,3,4,3,2,1,0,1,2,4,3,2,1,0,1,2,3,4,5,4,5,6,5,4]
lab31 = [1,16,19,22,100,109,111,112,113,118,119,242,245,248,249,251,262,269,273,321,324,396,397,398,489,490,493,494,496,547,548,553,572,574,577]
lab32 = [11,10,9,8,9,8,9,10,11,10,9,8,7,6,7,8,7,8,9,8,7,6,5,4,3,2,3,4,5,6,7,8,9,10,11]
diff_number=[0,1,0,-1,1,-1,0,-1,-1,1,-1,1,-1,1,-1,0]
people_number_frame=[0,36,37,44,54,77,166,189,214,225,658,678,689,770,775,795]
def track_result(dir_,a,b):
    with open(os.path.join('track_top',dir_),'w+') as f:
        for i in range(len(a)):
            text = 'frame: {}, num: {}\n'.format(a[i], b[i])
            f.write(text)
    f.close()
track_result('1.txt',people_number_frame,diff_number)
people_number_frame=[14,26,27,28,32,45,50,60,113,148,158,164,173,213,256,259,262,278,281,283]
diff_number=[0,-1,-1,0,1,1,1,1,-1,-1,0,-1,4,-1,-2,-1,1,0,2,0]
track_result('2.txt',people_number_frame,diff_number)
people_number_frame=[0,16,19,22,100,109,111,112,113,119,242,245,251,262,269,273,321,397,398,490,496,548,553,572,574,577,720,759,836]
diff_number=[0,-1,-1,-1,1,-1,1,1,1,0,-1,-1,2,-1,1,1,-1,-2,-1,-2,3,2,1,1,1,1,-1,1,0]
track_result('3.txt',people_number_frame,diff_number)
def labels(a,b,num,dir_):
    lab = np.zeros(num)
    log=[]
    for i in range(len(a)-1):
        lab[a[i]-1:a[i+1]] = b[i]
    lab[a[i+1] :num] = b[i+1]
    for i in range(len((lab))):
        text = 'frame: {}, num: {}'.format(i+1, int(lab[i]))
        log.append(text + '\n')
    with open(('gt/'+dir_), 'w+') as f:
        f.writelines(log)
    f.close()
    return lab

labels(lab11,lab12,796,dir_='1.txt')
labels(lab21,lab22,917,dir_='2.txt')
labels(lab31,lab32,837,dir_='3.txt')