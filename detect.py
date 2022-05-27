import argparse
import cv2
from yolo_detect import yolov5
from read_plate import ocr

class detection:
    def __init__(self):
        self.model_path = 'weights/licence_yv5m.pt'
        self.device = 'cpu'
        self.args = parse_args()
        self.image_file = self.args.image_file
        self.detect = yolov5()
        self.num = ocr()
   
    def main(self):     
        img = cv2.imread(self.image_file)
        img2=img.copy()
        #detect = yolov5()
        pt = self.detect.load_model(self.model_path,self.device)
        frame,box1=self.detect.detection(img,pt,img2)
        if len(box1) != 0:
            text = self.num.detect_ocr('cropped_pics/licence_1.png')
            print("DETECTED NUMEBER PLATE IS : ",text)
        else:
            print("NO NUMBER PLATE DETECTED")
        cv2.imshow('Detection', frame)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
        
def parse_args():
    parser = argparse.ArgumentParser(description="gets image file")
    
    parser.add_argument("--image",
                        dest="image_file",
                        required=True,
                        help="Image file to detect")
    return parser.parse_args()

if __name__ == '__main__':
    det = detection()
    det.main()
   
