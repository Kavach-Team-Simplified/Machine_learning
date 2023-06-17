import numpy as np
import cv2
from scipy.spatial import distance as dist
import argparse
import imutils
import cv2
import os
import time
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import base64

MIN_CONF = 0.3
NMS_THRESH = 0.3
MIN_DISTANCE = 50

def base64_to_image(base64_string):
    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

app = Flask(__name__, static_folder="./templates/static")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, async_mode="eventlet")

def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    centroids = []
    confidences = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personIdx and confidence > MIN_CONF:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)
    
    return results

@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})

@socketio.on("image")
def receive_image(image):
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]
    image = base64_to_image(image)
    frame = imutils.resize(image, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))    
    print(len(results))
    a = len(results)
    print(a)
    if a<3:
        
        emit("processed_text", str(a))
    else:
        emit("processed_text", "Crowd Detection: "+str(a))
    

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
    args = vars(ap.parse_args(["--input", r"C:/Kavach/ML/Models/CrowdDetection/test2.mp4", "--output", "my_output.avi", "--display", "1"]))
    labelsPath = os.path.sep.join([r"C:/Kavach/ML/Models/CrowdDetection/yolo-coco/coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = os.path.sep.join([r"C:/Kavach/ML/Models/CrowdDetection/yolo-coco/yolov3.weights"])
    configPath = os.path.sep.join([r"C:/Kavach/ML/Models/CrowdDetection/yolo-coco/yolov3.cfg"])
    socketio.run(app, debug=True)
