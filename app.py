from wsgiref import simple_server
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS,cross_origin
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
croppedImagepath = "cropped_pics\licence_1.png"
outputImagepath = "output\output_1.png"

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

@application.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@application.route("/predict", methods=["POST"])
@cross_origin()
def getPrediction():
    inpImage = request.json['image']
    decodeImageIntoBase64(inpImage, imagePath)
    frame = cv2.imread(imagePath)
    frame2 = frame.copy()
    pt = clApp.numberPlateObj.load_model(clApp.modelArg,clApp.device)
    try:
        frame,box = clApp.numberPlateObj.detection(frame,pt,frame2)
        if len(box) != 0:
            #opencodedbase64 = encodeImageIntoBase64(croppedImagepath)
            #ig = str(encodedCroppedImageStr)
            #ik = ig.replace('b\'', '')
            text = clApp.detocr.detect_ocr(croppedImagepath)
            clApp.numberPlateObj.draw_num(frame,text)
            opencodedbase64 = encodeImageIntoBase64(outputImagepath)
            responseDict = {"image": opencodedbase64.decode('utf-8'), "numberPlateVal":text}
            return jsonify(responseDict)
        else:
            responseDict = {"image": "Unknown", "numberPlateVal": "Unknown"}
                # responseList.append(responseDict)
                # print(responseDict)
                # convert to json data
            return jsonify(responseDict)
            # print(jsonStr.decode())
    
    except Exception as e:
        print(e)
    responseDict = {"image": "Unknown", "numberPlateVal": "Unknown"}
    # responseList.append(responseDict)
    # print(responseDict)
    # convert to json data
    return jsonify(responseDict)

# port = int(os.getenv("PORT"))
if __name__ == '__main__':
    clApp = ClientApp()
    # host = "127.0.0.1"
    # host = '127.0.0.1'
    # port = 5000
    # httpd = simple_server.make_server(host, port, application)
    # print("Serving on %s %d" % (host, port))
    # httpd.serve_forever()
    application.run()