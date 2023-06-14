import numpy as np
import cv2
from scipy.spatial import distance as dist
import argparse
import imutils
import cv2
import os
import time
MIN_CONF = 0.3
NMS_THRESH = 0.3
MIN_DISTANCE = 50


def detect_people(frame, net, ln, personIdx=0):

	(H, W) = frame.shape[:2]
	results = []
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
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


# @app.route("/")
# def form():
#     print("------------start-----------")
#     return render_template("index.html")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,help="whether or not output frame should be displayed")
args = vars(ap.parse_args(["--input",r"C:/Kavach/ML/Models/CrowdDetection/test2.mp4","--output","my_output.avi","--display","1"]))
labelsPath = os.path.sep.join([r"C:/Kavach/ML/Models/CrowdDetection/yolo-coco/coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([r"C:/Kavach/ML/Models/CrowdDetection/yolo-coco/yolov3.weights"])
configPath = os.path.sep.join([r"C:/Kavach/ML/Models/CrowdDetection/yolo-coco/yolov3.cfg"])
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

while True:
	time.sleep(10)
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,personIdx=LABELS.index("person"))
	violate = set()
	if len(results) >= 2:
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				if D[i, j] < MIN_DISTANCE:
					violate.add(i)
					violate.add(j)
	print(len(violate))