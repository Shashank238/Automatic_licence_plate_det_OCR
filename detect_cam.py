from pathlib import Path
import numpy as np
from imutils.video import VideoStream
import cv2
import torch
import datetime
import time
from datetime import date
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS
from utils.augmentations import letterbox
from utils.general import (LOGGER, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device
from utils.detect_utils import draw_box_on_image, draw_text_on_image
from utils.detect_utils import load_cam , crop_img


if __name__ == '__main__':
    weights = 'weights/licence_yv5m.pt'
    device = 'cpu'
    dnn = False
    data = 'data/data.yaml'
    imgsz = (640,640)
    half = False
    #path = 'car.png'
    conf_thres = 0.30
    iou_thres = 0.5
    classes = 1
    agnostic_nms= False
    max_det = 1000
    score_tresh = 0.30
    num_face_detect = 1

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    vs = cv2.VideoCapture(0)
    

    start_time = datetime.datetime.now()
    num_frames = 0
    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

    try:
        while True:
            time.sleep(0.7)
            _,frame = vs.read()

            frame = np.array(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if im_height == None:
                im_height, im_width = frame.shape[:2]
             
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame2=frame.copy()
            except:
                print("Error converting to RGB")
            
            dataset = load_cam(frame, img_size=imgsz, stride=stride, auto=pt)
            bs = 1
            model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
            dt, seen = [0.0, 0.0, 0.0], 0
            im = torch.from_numpy(dataset).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]
            
            pred = model(im)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.40, classes=None, max_det=300)

            my_dict = {"bbox":[],"scores":[],"classes":[]}
            cnt=0
            for i, det in enumerate(pred):
                # per image  
                gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh   
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()
                    det = det.tolist()
                    for *xyxy, conf, cls in det:
                        my_dict["bbox"].append(xyxy)
                        my_dict["scores"].append(conf)
                        my_dict["classes"].append(cls)
                        
            boxes =np.array(my_dict['bbox'])
            scores =np.array(my_dict['scores'])
            classes =np.array(my_dict['classes'])
            
            cnt,box= draw_box_on_image(num_face_detect, score_tresh, scores, boxes, classes, im_width=frame.shape[1], im_height=frame.shape[0],image_np= frame)

            cv2.putText(frame, str(cnt),
                    (int(frame.shape[1] * 0.95), int(frame.shape[0] * 0.9 + 30 )),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
            cv2.putText(frame, ' ' + str(cnt),
                        (int(im_width * 0.85), int(im_height * 0.9 + 30 )),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if len(box):
                crop_img(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB),box,'cropped_pics')
                time.sleep(.25)
                #cv2.destroyAllWindows()
                # vs.release()
                #break
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                vs.release()
                break

        print("Average FPS: ", str("{0:.2f}".format(fps)))
    except KeyboardInterrupt:
        print("Average FPS: ", str("{0:.2f}".format(fps)))
