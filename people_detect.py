import argparse
import cv2
from utils.detect import Detector
from yolox.data.datasets import COCO_CLASSES,Train_CLASSES,Filter_CLASSES
import torch
from utils.filter import filter,head_filter
from utils.joint_detect import joint_de
from yolox.utils import  vis
import os
import time
from utils.visual import visual

def imageflow_demo(head_predictor,other_predictor,args):
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
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            count+=1
            if count%5 != 0:
                continue

            head_info = head_predictor.detect(frame.copy())

            other_info = other_predictor.detect(frame.copy())
            
            # false head filter
            # if head_info['box_nums']>0:
            #     head_info = head_filter(head_info)

            #  jointdet
            if head_info['box_nums']*other_info['box_nums']>0:

                other_info = filter(other_info)
                result_info = joint_de(head_info, other_info)


            else:
                result_info = head_info

            result_img = visual(frame,result_info,Train_CLASSES,color=(0,255,255))

            text = 'frame: {}, num: {}'.format(count//5,result_info['box_nums'])
            cv2.putText(result_img, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0,0), 5)
            vid_writer.write(result_img)
            if count//5==829:
                cv2.imwrite(os.path.join(save_folder,'829.jpg'),result_img)
            print(text)
            text = 'frame: {}, num: {},bboxes: {}'.format(count // 5, result_info['box_nums'],result_info['boxes'])
            log.append(text+'\n')
        else:
            break
    with open(save_path.replace('.mp4','.txt'), 'w+') as f:
        f.writelines(log)
    f.close()

def image_demo(head_predictor,other_predictor,args):
    t1=time.time()
    raw_img = cv2.imread(args.path)

    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, args.path.split("/")[-1])

    head_info = head_predictor.detect(raw_img)
    other_info = other_predictor.detect(raw_img)
    t2=time.time()
    print(t2-t1)
    other_info = filter(other_info)
    result_info = joint_de(head_info, other_info)
    print(time.time()-t2)
    result_info['visual'] = vis(result_info['raw_img'].copy(), result_info['boxes'], result_info['scores'],
                              result_info['class_ids'], args.conf,
                              Train_CLASSES)
    cv2.imwrite(save_path,result_info['visual'])


if __name__=='__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument('-p', "--path", type=str, default="./1.mp4", help="choose a video")
    parser.add_argument('-m', "--model", type=str, default='yolox-s', help="choose a model")
    parser.add_argument("--camid", type=int, default=5, help="webcam demo camera id")
    parser.add_argument('-c','--ckpt',type=str,default='weights/yolox_s.pth',help="weight")
    parser.add_argument('-d', '--device', type=int, default='0', help='choose a gpu')
    parser.add_argument('-e', '--exp_files', type=str, default=None,
                        help="pls input your expriment description file")
    parser.add_argument('-f', '--conf', type=float, default=0.2,
                        help="sence test conf")
    parser.add_argument('--save_folder', type=str, default='result/total',
                        help="save path")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    #args.demo = 'video'
    args.nms=None
    #args.path = "/home/liup/Document/data/test/video/new/1.mp4"
    detector = Detector(args)
    #  head detector config
    args1= args
    args1.exp_files='exps/example/custom/yolox_s_head.py'
    args1.ckpt = 'weights/yolox_s_head.pth'


    args1.model=None

    args1.conf = 0.32   # head test conf
    args1.nms=0.5
    detector1 = Detector(args1, classes=Train_CLASSES)

    if args.demo=='image':
        image_demo(detector1,detector,args)
    if args.demo=='video':
        imageflow_demo(detector1,detector,args)
