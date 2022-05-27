import numpy as np
from pathlib import Path
from threading import Thread
from datetime import datetime
import cv2
import os
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from utils.augmentations import letterbox
from utils.datasets import IMG_FORMATS

def load_single_image(path, img_size=640, stride=32, auto=True):
    p = str(Path(path).resolve())
    files = [p]
    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    ni = len(images)
    img0 = cv2.imread(path)  # BGR
    img = letterbox(img0, img_size, stride=stride, auto=auto)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return  img0, img

def load_cam (frame,img_size = 640, stride = 32, auto = True):
    img = letterbox(frame, img_size, stride=stride, auto=auto)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return img

def draw_box_on_image(num_face_detect, score_thresh, scores, boxes, classes,  image_np):

    face_cnt = 0
    color = None
    color0 = (34,255,76)
    box =[]
    if len(boxes):
        for i in range(num_face_detect):
            if (scores[i] > score_thresh):
                if classes[i] == 0:
                    id = 'licence'
                    color = color0
                    (left, right, top, bottom) = (boxes[i][1] , boxes[i][3] ,
                                            boxes[i][0] , boxes[i][2] )
                    b = [int(top), int(left),int(bottom), int(right)]
                    box.append(b)
                    p1 = (int(top), int(left))
                    p2 = (int(bottom), int(right))
                    
                    face_cnt = face_cnt + 1
                    cv2.rectangle(image_np,p1,p2,color,3,1)

                    cv2.putText(image_np, 'licence' + str(i) , (int(top), int(left) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    cv2.putText(image_np, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                            (int(top), int(left) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return face_cnt,box
    else:
        return face_cnt,box

def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (77, 255, 9), 2)

def crop_img(frame,box,out_dir='cropped_pics'):
    for i in range(len(box)):
        xmin, ymin, xmax, ymax = box[i]
        cropped_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
        #now = date.today()
        img_name = "licence_1.png"
        img_path = os.path.join(out_dir, img_name)
        # save image
        cv2.imwrite(img_path, cropped_img)