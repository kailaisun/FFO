import numpy as np
def joint_tr(data_track, data_top,area_num):
    frame_track = data_track.frame.copy()
    frame_top = data_top.frame.copy()
    #area_num = data_track.area_num.copy()
    num_track = data_track.num.copy()
    num_top = data_top.num.copy()
    joint_frame = []
    joint_num = []
    no_confirm = []
    for i in frame_track:
        if area_num[i - 1] == 0:
            no_confirm.append(i)

    for i in range(len(frame_top)):
        frame = [i for i in [frame_top[i], frame_top[i] -1, frame_top[i] +1] if i in frame_track]

        if len(frame) > 0:
            index_ = frame_track.index(frame[0])



            if abs(num_top[i]) > abs(num_track[index_]) & (num_top[i] * num_track[index_] > 0):

                if (frame[0] - frame_track[index_ - 1] < 4):  #(怎么描述，以门口为主)

                    num_track[index_] = num_top[i] - num_track[index_]

                else:

                    num_track[index_] = num_top[i]
            elif num_top[i] * num_track[index_] < 0:

                if area_num[frame[0] - 2] - area_num[frame[0] - 1] > 0:
                    num_track[index_] = max(num_track[index_], num_top[i])
                else:
                    num_track[index_] = min(num_track[index_], num_top[i])

            joint_frame.append(frame[0])
            joint_num.append(num_track[index_])

            del frame_track[index_]
            del num_track[index_]



        elif area_num[frame_top[i] - 1] > 0:
            joint_frame.append(frame_top[i])
            joint_num.append(num_top[i])



    for i in frame_track:
        if area_num[i-1] < 3 :

            joint_frame.append(i)
            joint_num.append(num_track[frame_track.index(i)])
    frame_ = np.array(joint_frame)
    num = np.array(joint_num)
    index = frame_.argsort()
    frame_ = frame_[index]
    num = num[index]
    return frame_, num
