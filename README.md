ATM Face Recognition
This project is a face recognition system designed for ATM authentication. It uses OpenCV for face detection and recognition and stores the user data (name and ID) along with the captured face images. The system allows users to enroll their face and associate it with their user ID for future recognition.
Features:
Real-time face detection using Haar Cascades.
Capture and save face images for training.
User data (name and enrollment ID) stored in a CSV file.
Easy enrollment process with a webcam.
Requirements:
Python 3.x
OpenCV
CSV (for saving user data)
Install Dependencies:
pip install opencv-python
pip install opencv-contrib-python
Usage:
Step 1: Prepare Haar Cascade Classifier:
Make sure to download the haarcascade_frontalface_default.xml file from the official OpenCV repository or ensure it is available at the correct path.
Step 2: Running the Program:
Clone or download this repository.
Replace the paths in the code to match your system:
dataset_folder: Folder where the captured images will be stored.
csv_filename: Path where the user data (ID, name) will be saved in CSV format.
haarcascade_frontalface_default.xml: Path to the Haar Cascade XML file for face detection.
Enroll User:
User ID: Enter the unique ID associated with the user.
User Name: Enter the name of the user.
The system will then start the webcam and begin detecting faces. Once a face is detected, it will be saved as an image along with the user's ID and name. The program will capture 190 samples of the user's face and store them in the specified folder.
