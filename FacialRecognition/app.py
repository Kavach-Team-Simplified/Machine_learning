from keras_facenet import FaceNet
from flask import Flask, request, render_template
from flask_cors import CORS
import numpy as np
import os
import json
import cv2

app = Flask(__name__)
CORS(app)
app.config["CORS_HEADERS"] = "application/json"


@app.route("/")
def form():
    print("------------start-----------")
    return render_template("index.html")


embedder = FaceNet()


@app.route("/register", methods=["POST"])
def register1():
    print("Here------------------------")

    # Getting Username and filename from the form i.e. Index.html

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # Save the captured frame to a temporary file
    img_filename = "registration_image.jpg"
    cv2.imwrite(img_filename, frame)
    cap.release()

    # img1 = request.files["image_file"].filename
    user_name = request.form.get("User_name")
    print(user_name)
    # image_path = os.path.abspath(img1)
    # print(image_path)

    with open("C:\Kavach\ML\Models\FacialRecognition\database.json", "r") as file:
        data = json.load(file)

    # Check if the username already exists
    if user_name in data:
        return "User name already exists!"
    
    # Using Facenet to Extract Image data
    a = embedder.extract(img_filename, threshold=0.95)[0]
    a1 = a['box']
    a2 = a['confidence']
    a3 = a['keypoints']
    a4 = a['embedding'].tolist()

    # Add the new username to the JSON data
    data[user_name] = {'box':a1,'confidence':a2,'keypoints':a3,'embedding':a4}


    # Write the updated JSON data back to the file
    with open("database.json", "w") as file:
        json.dump(data, file)
    os.remove(img_filename)
    return "Registration successful of "+user_name


@app.route("/verify", methods=["POST"])
def verify1():
    # Open the camera and capture a frame
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # Save the captured frame to a temporary file
    img_filename = "verification_image.jpg"
    cv2.imwrite(img_filename, frame)
    cap.release()
    detections = embedder.extract(img_filename, threshold=0.95)[0]
    embedding1 = detections["embedding"]
    with open("C:\Kavach\ML\Models\FacialRecognition\database.json", "r") as file:
        data = json.load(file)
    i=0
    k=[]
    print(len(data))
    for j in data:
        embedding2 = np.array(data[j]["embedding"])
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        print(similarity)
        k.append([j,similarity])
    k1 = sorted(k,reverse=True,key = lambda x:x[1])
    print(k1)
    if k1[0][1] >= 0.750:
        os.remove(img_filename)
        return "Welcome "+k1[0][0]
    else:
        os.remove(img_filename)
        return "Denied"

if __name__ == "__main__":
    app.run(debug=True)
