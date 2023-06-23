# Graduation Back-End Application


[![Flask](https://img.shields.io/badge/Flask-2.3-blue.svg?style=for-the-badge)](https://pypi.org/project/Flask/)
[![Docker](https://img.shields.io/badge/Docker-20.10.7-blue.svg?style=for-the-badge)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7-orange.svg?style=for-the-badge)](https://opencv.org/)
[![Gunicorn](https://img.shields.io/badge/Gunicorn-20.1.0-red.svg?style=for-the-badge)](https://gunicorn.org/)
[![GCP](https://img.shields.io/badge/GCP-Google%20Cloud%20Platform-yellow.svg?style=for-the-badge)](https://cloud.google.com/)

This repository contains the back-end API for classifying violence types in videos using a Keras model. It is built using Flask.

## Prerequisites

- Python 3.9
- pip (Python package installer)

## Installation

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/moharamfatema/grad-be.git
   ```

2. Navigate to the project directory:
   ```
   cd grad-be
   ```

3. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   ```

4. Activate the virtual environment:
   - For Windows:
     ```
     venv\Scripts\activate
     ```
   - For macOS/Linux:
     ```
     source venv/bin/activate
     ```

5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Change directory to the source directory:
   ```
   cd src
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. The API server will start running locally on your machine.

## API Endpoints

The following endpoints are available for accessing the API:

- `POST /predict`: Upload a video file and specify the prediction type [`binary`,`multi-class`] and receive the predicted violence types.

## Tip: Send requests using ThunderClient

To use the Thunder Client extension in VS Code to upload a video using form data, follow these steps:

1. Install the Thunder Client extension in VS Code if you haven't already. You can find it in the VS Code extensions marketplace.

2. Open the Thunder Client extension by clicking on the Thunderbolt icon in the sidebar or by using the shortcut Ctrl+Alt+P (Cmd+Option+P on macOS) and typing "Thunder Client".

3. Create a new Thunder Client request by clicking on the "+" button or using the shortcut Ctrl+N (Cmd+N on macOS).

4. Set the request method to "POST" and enter the URL of the endpoint where you want to upload the video.

5. Under the "Body" tab, select "Form Data" as the request body type.

6. Click on the "+" button to add a new form field. Set the "Key" to the name of the field where the video should be uploaded. For example, enter "video" as the key.

7. Set the "Value" of the form field to the path of the video file on your local machine. You can use the file picker icon to browse and select the video file.

8. Click on the "+" button to add a new form field. Set the "Key" to "prediction_type" and add the desired value.

8. If needed, you can add more form fields by clicking on the "+" button and repeating steps 6 and 7.

9. Click on the "Send" button to send the request and upload the video using form data.

10. You will receive the response from the server, which you can view in the Thunder Client interface.
