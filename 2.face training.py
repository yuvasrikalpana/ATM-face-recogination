import cv2
import numpy as np
from PIL import Image
import os
 
path = 'C:/Users/user name/Documents/Face Recoganization/TrainingImage'
recognizer = cv2.face.LBPHFaceRecognizer_create()
global detector
detector = cv2.CascadeClassifier("C:/Users/user name/Documents/Face Recoganization/haarcascade_frontalface_default.xml")
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]    
    faceSamples=[]
    Ids = []
    for imagePath in imagePaths:
        PilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(PilImage,'uint8')
 
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(imageNp)
 
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h,x:x + w])
            Ids.append(Id)
    return faceSamples,Ids
 
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,Ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(Ids))
recognizer.write('C:/Users/user name/Documents/Face Recoganization/trainer/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(Ids))))
 

