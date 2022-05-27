import torch
import numpy as np
import cv2
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS
from utils.general import (LOGGER, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords)
from utils.torch_utils import select_device
from utils.detect_utils import load_cam,draw_box_on_image,crop_img,draw_text_on_image

class yolov5:
    
    def __init__(self):
        
        self.model = None
        self.device = None
        self.dnn = False
        self.data = None
        self.imgsz = (640,640)
        self.stride = 64
        self.half = False
        self.conf_thres = 0.30
        self.iou_thres = 0.5
        self.classes = 1
        self.num_lic_detect = 1
        self.agnostic_nms= False
        self.max_det = 1000
        self.score_tresh = 0.30
        self.out_dir = 'cropped_pics'
        self.out_pic = 'output/output_1.png'
    
    def load_model(self,model_pth,device):
        self.device = select_device(device)
        self.model = DetectMultiBackend(model_pth, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride,  pt = self.model.stride, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        return pt
   
    def detection(self,frame,pt,frame2):        
        dataset = load_cam(frame, img_size=self.imgsz, stride=self.stride, auto=pt)
        bs = 1
        self.model.warmup(imgsz=(1 if pt else bs, 3, *self.imgsz))
        dt, seen = [0.0, 0.0, 0.0], 0
        im = torch.from_numpy(dataset).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.40, classes=None, max_det=300)

        my_dict = {"bbox":[],"scores":[],"classes":[]}
        for i, det in enumerate(pred):
            # per image    
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()
                det = det.tolist()
                for *xyxy, conf, cls in det:
                    my_dict["bbox"].append(xyxy)
                    my_dict["scores"].append(conf)
                    my_dict["classes"].append(cls)
                    
        boxes1 = np.array(my_dict['bbox'])
        scores1 = np.array(my_dict['scores'])
        classes1 = np.array(my_dict['classes'])
        _,box1= draw_box_on_image(self.num_lic_detect,self.score_tresh, scores1, boxes1, classes1,image_np= frame)
       # cv2.imwrite(self.out_pic,frame)
        #cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if len(box1):
            crop_img(frame2,box1,self.out_dir)
        return frame,box1

    def draw_num(self,frame,txt):
        draw_text_on_image(txt,frame)
        cv2.imwrite(self.out_pic,frame)

