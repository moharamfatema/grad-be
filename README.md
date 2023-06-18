# Graduation Back-End Application

This repository contains the back-end API for classifying violence types in videos using a Keras model. It is built using Flask.

## Prerequisites

- Python 3.x
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
