# Face Recognition Attendance System

A Flask-based app for automating attendance using facial recognition.

## Features

- Face registration and recognition.

- Attendance saved daily in CSV format.

- Web-based UI for user management and tracking.

- Dynamic model training with OpenCV and KNN.

## Requirements

- Python 3.x, Webcam.

- Install dependencies: pip install flask opencv-python-headless numpy pandas scikit-learn joblib

## Setup

- Clone the repo and navigate to the folder.

- Install dependencies: pip install -r requirements.txt

- Run the app: python app.py

- Open http://127.0.0.1:5000/ in a browser.

## Project Structure

- app.py: Main application file.

- Attendance/: Attendance CSVs.

- static/faces/: Stored user images.

- templates/: HTML files for UI.

- haarcascade_frontalface_default.xml: Face detection model.



