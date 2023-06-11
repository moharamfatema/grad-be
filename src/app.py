from flask import Flask, jsonify, request, flash, redirect, url_for
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import urllib.request as req
from app.modules.model.model import Predictor
from app.modules.process.process import video_to_array
from pathlib import Path

from logging import getLogger
from logging.config import dictConfig

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
                "datefmt": "%d-%m-%y %H:%M:%S",
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

UPLOAD_FOLDER = "tmp/videos"
# allow videos
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}


predictor = Predictor()
input_shape = predictor.get_input_shape()


def prediction_from_video(video_path):
    getLogger().debug(f"prediction_from_video: {video_path.name}")
    getLogger().debug(f"input_shape: {input_shape}")
    array = video_to_array(
        video_path, resize=input_shape[2:-1], max_frames=input_shape[1]
    )
    if array is not None:
        prediction = predictor.predict(array)
        return prediction
    else:
        return None


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_request(request):

    if "video" not in request.files:
        return False, "No file found"

    file = request.files["video"]
    # app.logger.debug(request.files)

    if not file:
        return False, "No file found"

    if file.filename == "":
        return False, "No selected file"

    if not allowed_file(file.filename):
        return False, "Invalid file type"

    return True, file


@app.route("/")
def health_check():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    # validate request
    valid, ret = validate_request(request)
    if not valid:
        return jsonify({"status": "400", "message": ret})

    file = ret

    # save video to tmp
    filename = secure_filename(file.filename)
    file.save(Path(app.config["UPLOAD_FOLDER"]).joinpath(filename))

    # app.logger.debug(file.filename)
    # perform prediction
    prediction = None
    resp = None
    try:
        prediction = prediction_from_video(
            Path(app.config["UPLOAD_FOLDER"]).joinpath(filename)
        )
        if prediction is None:
            resp = jsonify({"status": "400", "message": "Video processing failed"})
        else:
            resp = jsonify({"status": "200", "prediction": prediction})
    except Exception as e:
        app.logger.error(e)
        resp = jsonify({"status": "400", "message": "Video processing failed"})
    finally:
        # delete the video
        Path(app.config["UPLOAD_FOLDER"]).joinpath(filename).unlink()
        pass

    return resp


if __name__ == "__main__":
    app.run(port=8080, debug=True)
