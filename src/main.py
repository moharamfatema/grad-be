"""
App entry point, contains the flask app and the prediction endpoint

functions:
    prediction_from_video: perform prediction on the given video file
    allowed_file: check if the file is allowed
    validate_request: validate the request
    predict: prediction endpoint
    health_check: health check endpoint
"""
from pathlib import Path

from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from app.modules.util.log import log
from app.modules.model.model import Predictor
from app.modules.process.process import video_to_array

BAD_REQUEST="Bad Request"
OK = "ok"
UPLOAD_FOLDER = "tmp/videos"
PREDICTION_TYPES = {"binary", "multi-class"}
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}  # allow videos

Path.exists(Path(UPLOAD_FOLDER)) or Path(UPLOAD_FOLDER).mkdir(parents=True)

predictor = Predictor()

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def prediction_from_video(video_path: Path, predictor:Predictor = predictor, prediction_type: str = "binary"):
    """
    Perform prediction on the given video file

    Args:
        video_path (Path): Path to the video file
        prediction_type (str, optional): Type of prediction to perform.

    Returns:
        np.ndarray: Prediction array
    """
    input_shape = predictor.get_input_shape()

    log.debug("prediction_from_video: %s", video_path.name)
    log.debug("input_shape: %s", input_shape)
    array = video_to_array(
        video_path, resize=input_shape[2:-1], max_frames=input_shape[1]
    )
    if array is None:
        return None
    if prediction_type == "binary":
        prediction = predictor.predict_binary(array)
    elif prediction_type == "multi-class":
        prediction = predictor.predict_multi(array)
    return prediction


def allowed_file(filename: str):
    """Check if the file is allowed
    Args:
        filename (str): Name of the file
    Returns:
        bool: True if allowed, False otherwise
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def validate_request(req: request):
    """Validate the request
    Performed validations:
    - file exists
    - file is not empty
    - file is of allowed type
    - prediction type is valid
    - prediction type is [binary, multi-class]

    Args:
        req (flask.request): Flask request object

    Returns:
        bool: True if valid, False otherwise
        str: Error message if not valid
    """

    if "video" not in req.files:
        return False, "No file found"

    file = req.files["video"]

    if not file or file.filename == "":
        return False, "No file found"

    if not allowed_file(file.filename):
        return False, "Invalid file type"

    # request includes prediction type and is [binary, multi-class]
    if "prediction_type" not in req.form:
        return False, "No prediction type found"

    if req.form["prediction_type"] not in PREDICTION_TYPES:
        return False, "Invalid prediction type"

    return True, None


@app.route("/")
def health_check():
    """Health check endpoint"""
    return jsonify({"status": OK})


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    """Prediction endpoint"""
    # validate request
    valid, ret = validate_request(request)
    if not valid:
        return jsonify({"status": BAD_REQUEST, "message": ret}), 400

    file = request.files["video"]

    # save video to tmp
    filename = secure_filename(file.filename)
    file.save(Path(app.config["UPLOAD_FOLDER"]).joinpath(filename))

    # perform prediction
    prediction = None
    resp = None
    try:
        prediction = prediction_from_video(
            video_path=Path(app.config["UPLOAD_FOLDER"]).joinpath(filename),
            prediction_type=request.form["prediction_type"],
        )
        if prediction is None:
            resp = jsonify(
                {"status": BAD_REQUEST, "message": "Video processing failed"}
            ), 400
        else:
            resp = jsonify({"status": OK, "prediction": prediction})
    except Exception as exp:
        log.error(exp)
        resp = jsonify({"status": BAD_REQUEST, "message": "Video processing failed"})
    finally:
        # delete the video
        Path(app.config["UPLOAD_FOLDER"]).joinpath(filename).unlink()

    return resp

if __name__ == "__main__":
    app.run(port=8080, debug=True)
else:
    gunicorn_app = app
