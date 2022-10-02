from utils.visual import plt_sq
from utils.data_load import read_num,read_detect
import os
class data_sq:
    def __init__(self):
        self.sq_num = None
        self.name = None

if __name__=='__main__':


    for i in range(3):
        sq_data = {}
        dir_ = str(i+1)+'.txt'
        sq_data[1] = data_sq()
        sq_data[1].sq_num = read_num(os.path.join('gt', dir_))
        sq_data[1].name = 'truth'
        sq_data[2] = data_sq()
        sq_data[2].sq_num = read_detect(os.path.join('result/detect', dir_))
        sq_data[2].name = 'detect'
        # sq_data[3] = data_sq()
        # sq_data[3].sq_num = read_num(os.path.join('result/deepsort', dir_))
        # sq_data[3].name = 'deepsort'
        sq_data[4] = data_sq()
        sq_data[4].sq_num = read_num(os.path.join('result/joint_top', dir_))
        sq_data[4].name = 'our'
        plt_sq(sq_data, title='result/plot/test'+str(i+1))