from keras_facenet import FaceNet
from flask import Flask, request,render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS']='application/json'

@app.route('/')
def form():
    return render_template('index.html')

embedder = FaceNet()
database = {}

@app.route('/register',methods = ['POST'])
def register1():
    img1 = request.files["image_file"]
    img = img1.filename
    database["Swayam"] = embedder.extract(img, threshold=0.95)
    if len(database)>0:
        return render_template('home.html', prediction_text="Verification sucess")
    else:
        return render_template('home.html', prediction_text="Failed to Register")


@app.route('/verify',methods = ['POST'])
def verify1():
    img = request.files["image_file1"]
    detections = embedder.extract(img, threshold=0.95)
    embedding1 = detections[0]['embedding']
    for i in database:
        embedding2 = database[i][0]['embedding']
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        if similarity >= 0.800:
            return render_template('home.html', prediction_text="Welcome "+i)
        else:
            return render_template('home.html', prediction_text="Denied")

if __name__ == "__main__":
    app.run(debug=True)