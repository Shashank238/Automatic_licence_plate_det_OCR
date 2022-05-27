from PaddleOCR.paddleocr import PaddleOCR
import numpy as np
import cv2
import re
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
class ocr:
    def __init__(self):
        self.ocr =PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
    def detect_ocr(self,img_path):
        result = self.ocr.ocr(img_path,cls=True)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        txt= []
        img = cv2.imread(img_path)
        height, width,_ = img.shape
        for i in range(len(boxes)):
            n=np.array(boxes[i]).astype(np.float32)
            #print(n)
            x,y,w,h = cv2.boundingRect(n)
            if height / float(h) > 5.8: continue
            # ratio = h / float(w)
            # # if height to width ratio is less than 1.5 skip
            # if ratio < 0.5: continue
            # area = h * w
            # # if area is less than 100 pixels skip
            # if area < 100: continue
            txt.append(re.sub('[^A-Za-z0-9]+', '', txts[i]))
            # box = np.array([rect[1],rect[0],rect[1]+rect[3],rect[0]+rect[2]])
            # ar = (box[2] - box[0]) * (box[3] - box[1])
            # if ar > max_area:
            #     max_area = ar
        text = ''.join([str(item) for item in txt])
        #text = re.sub('[^A-Za-z0-9]+', '', text)

        return text
