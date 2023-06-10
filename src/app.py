from flask import Flask, jsonify, request, flash, redirect, url_for
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import urllib.request as req
from app.modules.model.model import Predictor
from app.modules.process.process import video_to_array
from pathlib import Path

from logging.config import dictConfig

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
                 "datefmt": "%B %d, %Y %H:%M:%S %Z",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            }
        },
        "file": {
                "class": "logging.FileHandler",
                "filename": "flask.log",
                "formatter": "default",
            },
        "root": {"level": "DEBUG", "handlers": ["console"]},

    }
)

UPLOAD_FOLDER = 'tmp/videos'
# allow videos
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}


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
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def health_check():
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    def bad_request(msg):
        return jsonify({'status':'400','message':msg})

    if 'video' not in request.files:
        return bad_request('No file found')
    file = request.files['video']
    app.logger.debug(request.files)
    if not file:
        return bad_request('No file found')
    if file.filename == '':
        return bad_request('No selected file')
    if not allowed_file(file.filename):
        return bad_request('Invalid file type')
    # prediction = prediction_from_video(vid)
    # save video to tmp
    filename = secure_filename(file.filename)
    file.save(Path(app.config['UPLOAD_FOLDER']).joinpath(filename))

    app.logger.debug(file.filename)
    prediction = f'received {file.filename}'
    if prediction is None:
        return jsonify({'status':'400','message':'Video processing failed'})

    return jsonify({'status':'200','prediction': prediction})


if __name__ == "__main__":
    app.run(port=8080,debug=True)
