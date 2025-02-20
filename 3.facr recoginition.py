
from tokenize import Name
import cv2
import numpy as np

import pandas as pd
#import csv

from pyparsing import col
filename = "C:/Users/user name/Documents/Face Recoganization/PeopleData/Data.csv"
global Enrollment
global Id
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/user name/Documents/Face Recoganization/trainer/trainer.yml')
harcascadePath = "C:/Users/user name/Documents/Face Recoganization/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        Id, conf = recognizer.predict(gray[y:y + h,x:x + w])
        if (conf <190):
            
          #  df = pd.read_csv('C:/Users/user name/Documents/Face Recoganization/PeopleData/Data.csv',index_col=1)
          #  df=(df.loc[df["Enrollment"] == Id])
          #  print(df)
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
            #cv2.rectangle(im, (x, y), (x + w, y + h), (0, 260, 0), 7)
            cv2.putText(im, str(Id), (x,y-40),font, 2, (255,255,255), 3)
            #cv2.putText(im, str(Id), (x + h, y), font, 1, (225, 225, 0), 4)
            cv2.imshow('im',im)
        else:
            Id = 'Unknown'
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 25, 255), 7)
            cv2.putText(im, str(Id), (x + h, y), font, 1, (0, 25, 255), 4)
            cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()