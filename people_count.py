from utils.track import Tracker
from utils.visual import vis_track,vis
import time,argparse,torch
import cv2,os
from utils.count import num_count,track
def imageflow_demo(tracker,args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('帧率：',fps)
    fps = 4
    count = 0
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    log = []
    num = 0
    paths = {}
    # entrance line
    line1 = [(1189,518),(1358,488)]
    line3 = [(1645,375),(1843,411)]
    area_1 = [(1086,181),(1370,616)]
    area_3 = [(1557,89),(1845,624)]
    area = area_1
    line = line1

    lt = (1189,0)
    rl = (1358,588)
    fpr = []
    num_arr = []
    area_num_arr = []
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            count+=1
            in_num = 0
            # if count%5 != 0:
            #     continue

            result_info = tracker.track(frame.copy())
            result_img = vis_track(frame.copy(),result_info['track'])
            output = result_info['track']
            for (x1,y1,x2,y2,id) in output:
                mid_x = (x1 + x2) // 2
                mid_y = y2
                if id not in paths.keys():
                    paths[id] = track((mid_x,mid_y))
                else:
                    paths[id].traj.append((mid_x,mid_y))
                    #path[id].update()
                paths[id].miss=20
            if len(output)>0:
                id_arr = output[:,-1]
                in_num,paths=num_count(id_arr, paths, line)
                num+=in_num

                mis_id = set(paths.keys()).difference(set(id_arr))
                for id in mis_id:
                    paths[id].miss-=1
                    if paths[id].miss<=0:
                        del paths[id]


            # from utils.box_tool import center
            # from utils.count import ifin
            # area_num = 0
            # if len(result_info['boxes'])>0:
            #     conter_point = center(result_info['boxes'])
            #
            #     for point in conter_point:
            #         if ifin(point,area[0],area[1]):
            #             area_num+=1
            if in_num!=0:
                fpr.append(count)
                num_arr.append(in_num)
                #area_num_arr.append(area_num)



            text = 'frame: {}, num: {}'.format(count,num)
            cv2.putText(result_img, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0,0), 4)
            cv2.line(result_img ,line[0], line[1], (0, 255, 255), 3)
            #cv2.rectangle(result_img,lt,rl,(0, 255, 255),2)
            vid_writer.write(result_img)
            print(text)
            log.append(str(count)+' frame: '+str(list(result_info['track']))+'\n')
        else:
            break
        with open(save_path.replace('.mp4', '.txt'), 'w+') as f:
            f.writelines(log)
        f.close()
        with open(save_path.replace('.mp4', '_.txt'), 'w+') as f:
            for i in range(len(fpr)):
                content = '{}, {} \n'.format(fpr[i],num_arr[i])
                f.write(content)
        f.close()
if __name__=='__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument('-p', "--path", type=str, default="./1.mp4", help="choose a video")
    parser.add_argument('-m', "--model", type=str, default='yolox-s', help="choose a model")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument('-c','--ckpt',type=str,default='weights/yolox_s.pth',help="weight")
    parser.add_argument('-d', '--device', type=int, default='5', help='choose a gpu')
    parser.add_argument('-e', '--exp_files', type=str, default=None,
                        help="pls input your expriment description file")
    parser.add_argument('-f', '--conf', type=float, default=0.4,
                        help="test conf")
    parser.add_argument('--nms', default=None,
                        help="nms")
    parser.add_argument('--save_folder', type=str, default='result/track',
                        help="save path")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    args.demo = 'video'
    #args.path = "/home/liup/Document/data/test/video/new/2.mp4"
    tracker = Tracker(args)
    imageflow_demo(tracker,args)