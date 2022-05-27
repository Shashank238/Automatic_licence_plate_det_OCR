from wsgiref import simple_server
from flask import Flask, request, Response
from flask_cors import CORS
import json
import cv2
import base64
import os
from yolo_detect import yolov5
from read_plate import ocr

application = Flask(__name__)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')
CORS(application)

inputFileName = "inputImage.jpg"
imagePath = "images/" + inputFileName
croppedImagepath = "cropped_pics/licence_1.png"

class ClientApp:
    def __init__(self):
        # modelArg = "datasets/experiment_faster_rcnn/2018_08_02/exported_model/frozen_inference_graph.pb"
        self.modelArg = "weights\licence_yv5m.pt"
        self.device = "cpu"
        filepath = "autoPartsMapping/partNumbers.xlsx"
        # self.regPartDetailsObj = ReadPartDetails(filepath)
        self.numberPlateObj = yolov5()
        self.detocr = ocr()

def decodeImageIntoBase64(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

@application.route("/predict", methods=["POST"])
def getPrediction():
    inpImage = request.json['image']
    decodeImageIntoBase64(inpImage, imagePath)
    frame = cv2.imread(imagePath)
    frame2 = frame.copy()
    pt = clApp.numberPlateObj.load_model(clApp.modelArg,clApp.device)
    try:
        _,box = clApp.numberPlateObj.detection(frame,pt,frame2)
        if len(box) != 0:
            encodedCroppedImageStr = encodeImageIntoBase64(croppedImagepath)
            ig = str(encodedCroppedImageStr)
            ik = ig.replace('b\'', '')
            text = clApp.detocr.detect_ocr(croppedImagepath)
            responseDict = {"base64Image": ik, "numberPlateVal": text}
            jsonStr = json.dumps(responseDict, ensure_ascii=False).encode('utf8')
                # print(jsonStr.decode())
            return Response(jsonStr.decode())
        else:
            responseDict = {"base64Image": "Unknown", "numberPlateVal": "Unknown"}
                # responseList.append(responseDict)
                # print(responseDict)
                # convert to json data
            jsonStr = json.dumps(responseDict, ensure_ascii=False).encode('utf8')
            # print(jsonStr.decode())
    
    except Exception as e:
        print(e)
    responseDict = {"base64Image": "Unknown", "numberPlateVal": "Unknown"}
    # responseList.append(responseDict)
    # print(responseDict)
    # convert to json data
    jsonStr = json.dumps(responseDict, ensure_ascii=False).encode('utf8')
    # print(jsonStr.decode())
    return Response(jsonStr.decode())

# port = int(os.getenv("PORT"))
if __name__ == '__main__':
    clApp = ClientApp()
    # host = "127.0.0.1"
    host = '127.0.0.1'
    port = 5000
    httpd = simple_server.make_server(host, port, application)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
    # application.run(host='0.0.0.0', port=port)