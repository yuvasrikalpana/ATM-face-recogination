import os
import cv2
import csv
import pandas

# Initialize webcam and set resolution
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Define paths
dataset_folder = 'C:/Users/user name/Documents/Face Recoganization/TrainingImage/'
csv_filename = r'C:/Users/admin/Documents/Downloads/user_data.csv'

# Load face detector (Haar cascade)
face_detector = cv2.CascadeClassifier('C:/Users/user name/Documents/Face Recoganization/haarcascade_frontalface_default.xml')

# Get user input
enrollment = input('\nEnter user ID and press <return> ==> ')
name = input('\nEnter user name and press <return> ==> ')

# Initialize sample number
sampleNum = 0

# Prepare the CSV file and check if it's empty (write header if needed)
if not os.path.isfile(csv_filename):
    with open(csv_filename, 'w', newline='') as csvFile:
        fieldnames = ['Enrollment', 'Name']
        writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
        writer.writeheader()

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_detector.detectMultiScale(gray, 5, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Increment sample number and save the image of the face
        sampleNum += 1
        cv2.imwrite(f"{dataset_folder}{name}.{enrollment}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow('Frame', img)

    # Wait for user to press 'q' to quit or if sampleNum exceeds 190
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif sampleNum >= 190:
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()

# Append user data to the CSV file
with open(csv_filename, 'a+', newline='') as csvFile:
    fieldnames = ['Enrollment', 'Name']
    writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
    
    # Writing the user's enrollment and name to the CSV file
    writer.writerow({'Enrollment': enrollment, 'Name': name})

print(f"Images Saved for Enrollment: {enrollment} Name: {name}")
