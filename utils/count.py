import math
import random

import numpy as np
from collections import deque
import  cv2
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def __distance_to_line(x, y, x1, y1, x2, y2):
    # from:http://blog.sina.com.cn/s/blog_5d5c80840101bnhw.html
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
    if (cross <= 0):
        return math.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1))

    d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    if (cross >= d2):
        return math.sqrt((x - x2) * (x - x2) + (y - y2) * (y - y2))

    r = cross / d2
    px = x1 + (x2 - x1) * r
    py = y1 + (y2 - y1) * r

    return math.sqrt((x - px) * (x - px) + (py - y1) * (py - y1))

def ifin(point,lt,rl):
    x1 = lt[0]
    x2 = rl[0]
    y1 = lt[1]
    y2 = rl[1]
    x = point[0]
    y = point[1]
    if x <= x1 or x >= x2 or y <= y1 or y >= y2: return False
    return True
class track:
    def __init__(self,point):
        self.traj = deque(maxlen=5)
        self.traj.append(point)
        #self.line0 = False
        self.line1 = False
        self.miss = 20
    def update(self):
        traj = deque(maxlen=5)
        traj.append(self.traj[-1])
        self.line1 = False
        self.miss = 20
        self.traj = traj

def num_count(id_arr, paths,  line1):
    in_num = 0
    #out_num = 0
    for id in id_arr:
        traj = paths[id].traj
        if len(traj)>1:
            # if intersect(traj[0],traj[-1],line0[0],line0[1]):
            #     paths[id].line0 = not paths[id].line0
            #
            #     paths[id].update()
            if intersect(traj[0],traj[-1],line1[0],line1[1]):
                #paths[id].line1 = not paths[id].line1
                #paths[id].update()

        #     if paths[id].logo != ifin(traj[-1],lt,rl):
        #         paths[id].logo = ifin(traj[-1],lt,rl)

                paths[id].line1 = True
                #paths[id].update()
            if paths[id].line1:
                tm = traj[-1][1]-traj[0][1]
                # dis0 = __distance_to_line(traj[0][0],traj[0][1],line0[0][0],line0[0][1],line0[1][0],line0[1][1])
                # dis1 = __distance_to_line(traj[0][0],traj[0][1], line1[0][0], line1[0][1], line1[1][0], line1[1][1])
                if tm<0:
                    in_num-=1
                else:
                    in_num+=1
                paths[id].update()

        # else:
        #     paths[id].logo = ifin(traj[-1], lt, rl)
                #paths[id] = track(traj[-1])
    return in_num,paths

def tradge(img,id_arr, paths):
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]



    for id in id_arr:
        traj = paths[id].traj
        if len(paths[id].traj)>1:
            arr=list(traj)
            clr = random.randint(0, 8)
            for point,point1 in zip(arr[0:-1],arr[1:]):

                #clr = random.randint(0,9)
                cv2.line(img, (point[0], point[1]), (point1[0], point1[1]), track_colors[clr], 2)
    ids = set(id_arr)
    mess_id = set(paths.keys()).difference(ids)
    for id in mess_id:
        paths[id].mess-=1
        if paths[id].mess<=0:
            del paths[id]
    return img,paths










