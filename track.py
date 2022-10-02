from utils.track import Tracker
from utils.visual import vis_track,vis
import time,argparse,torch
import cv2,os
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
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            count+=1
            if count%5 != 0:
                continue
            result_info = tracker.track(frame.copy())
            result_img = vis_track(frame.copy(),result_info['track'])


            text = 'frame: {}, num: {}'.format(count//5,len(result_info['track']))
            cv2.putText(result_img, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2)
            vid_writer.write(result_img)
            print(text)
            log.append(str(count//5)+' frame: '+str(list(result_info['track']))+'\n')
        else:
            break
        with open(save_path.replace('.avi', '.txt'), 'w+') as f:
            f.writelines(log)
        f.close()
if __name__=='__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser("YOLOX-Tracker Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument('-p', "--path", type=str, default="./test.jpg", help="choose a video")
    parser.add_argument('-m', "--model", type=str, default='yolox-s', help="choose a model")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument('-c','--ckpt',type=str,default='weights/yolox_s.pth',help="weight")
    parser.add_argument('-d', '--device', type=int, default='2', help='choose a gpu')
    parser.add_argument('-e', '--exp_files', type=str, default=None,
                        help="pls input your expriment description file")
    parser.add_argument('-f', '--conf', type=float, default=0.4,
                        help="test conf")
    parser.add_argument('--save_folder', type=str, default='result/track',
                        help="save path")
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    args.demo = 'video'
    args.path = "/home/liup/Document/data/test/video/new/1.avi"
    tracker = Tracker(args)
    imageflow_demo(tracker,args)