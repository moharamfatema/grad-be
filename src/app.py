from flask import Flask, jsonify, request
import urllib.request as req
from app.modules.model.model import Predictor
from app.modules.process.process import video_to_array
from pathlib import Path

predictor = Predictor()
input_shape = predictor.get_input_shape()

def prediction_from_video(video) -> str:
    array = video_to_array(video, resize=input_shape[1:], max_frames=input_shape[1])
    if array is not None:
        prediction = predictor.predict(array)
        return prediction
    else:
        return None

app = Flask(__name__)

@app.route('/')
def health_check():
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
def predict():
    vid = request.files.get('video')

    if not vid:
        return jsonify({'status':'400','message':'No video file found'})

    prediction = prediction_from_video(vid)
    if prediction is None:
        return jsonify({'status':'400','message':'Video processing failed'})

    return jsonify({'status':'200','prediction': prediction})


if __name__ == "__main__":
    app.run(port=8080,debug=True)
