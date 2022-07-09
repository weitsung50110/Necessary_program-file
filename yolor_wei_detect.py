import argparse #使程式執行時可帶參數
import os #用操作系統命令，來達成建立文件，刪除文件，查詢文件等
import platform #獲取系統資訊
import shutil #該模組擁有許多檔案（夾）操作的功能，包括複製、移動、重新命名、刪除等等
import time  #時間模組
from pathlib import Path #將各種檔案/資料夾相關的操作封裝在 Path 等類別之中，讓檔案/資料夾的操作更加物件導向
import cv2 #電腦視覺庫
import torch #基於 dynamic computation graph 讓架構能在訓練時改變。
import torch.backends.cudnn as cudnn #為整個網絡的每個卷積層搜索最適合它的捲積實現算法，進而實現網絡的加速
import csv #.csv庫
from numpy import random #但在此我們以演算法生成的資料(偽隨機, Pseudo Random)加上雜訊模擬為隨機資料

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                #results = {} #後來加的
                for c in det[:, -1].unique():  #c代表一個圖片裡面第幾個class 第一個類別是0, 第二個是1  以此類推
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

           #---------------------------------------------------------------------------------------------


                    print("\n"+"Class number : %d"%(c)) #c代表一個圖片裡面第幾個class 第一個類別是0, 第二個是1  以此類推
                    #print("\n"+"N : %g"%(n))
                    #print("\n"+"S : %s"%(s))
                    #print("\n"+"names[int(c)] : %s"%(names[int(c)]))
                    #print("--------------------------")
 
                    class_name = '%s' % (names[int(c)]) #class名稱
                    class_count = '%g' % (n)# 偵測圖片class統計數量

                    #寫入results字典裡
                    conf_count = 0 #預設第一個類別 因此是0, 第二個是1 以此類推
                    for *xyxy, conf, cls in det:
                        conf0 = '%.2f' % (conf)
                        if conf_count == c: #c是代表圖片中第幾個class的數字 若一樣就break 此舉是為了讓conf的值有辦法對應個別的類別
                            break
                        conf_count = conf_count+1 
                    
 
                    dic = {'image_name': os.path.split(path)[-1], 'class_name': class_name, 'class_count': class_count, 'conf0': conf0} #照片檔名 類別 數量 置信度


                    print("\n" + "dic_image_name : "+dic["image_name"])
                    print("dic_class_name : "+dic["class_name"])
                    print("dic_class_count : "+dic["class_count"])  #記得字典裡面必須要加雙引號~~~~
                    print("dic_conf0 : "+dic["conf0"])  #記得字典裡面必須要加雙引號~~~~

                    #照片檔名 我存在字典dic['image_name']裡面
                    #file_name=os.path.split(path)[-1]
                    #儲存.csv位置
                    save_csv_path = str(os.getcwd()) + '/' + 'inference' + '/' + 'parking_violation3_detect.csv'
                    #print(save_csv_path)
                    '''
                    #這邊不會用到
                    key = dic.keys()
                    class_key0=(list(key)[0]) #照片檔名
                    class_key1=(list(key)[1]) #照片類別
                    class_key2=(list(key)[2]) #類別的數量
                    class_key3=(list(key)[3]) #類別的置信度
                    print("class_key0 : "+class_key0)
                    print("class_key1 : "+class_key1)
                    print("class_key2 : "+class_key2)
                    print("class_key3 : "+class_key3)
                    '''

                    # 開啟輸出的 CSV 檔案
                    with open(save_csv_path, 'a+', encoding='utf-8') as csvfile:
                        if os.path.getsize(save_csv_path):
                            # 建立 CSV 檔寫入器
                            writer = csv.writer(csvfile)
                            # 寫入辨識資料
                            writer.writerow([dic["image_name"], dic["class_name"], dic["class_count"], dic["conf0"]])
                        else:
                            # 建立 CSV 檔寫入器
                            writer = csv.writer(csvfile)
                            # 寫入辨識標題
                            writer.writerow(["image_name", "class_name", "class_count", "confidence"])
                            # 寫入辨識資料
                            writer.writerow([dic["image_name"], dic["class_name"], dic["class_count"], dic["conf0"]])


                # ---------------------------------------------------------------------------------------------

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g' * 5 + '\n') % (cls, *xywh))  # label format


                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        #print("\n"+"LABEL : "+label)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    print(xywh) #預測出來的yolo標註點
                    print('%.2f' % (conf)) #預測出來的conf
                    #print("QQQQQQQQQQQQQQQQQQ")

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
