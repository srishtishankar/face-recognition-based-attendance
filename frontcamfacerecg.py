# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:06:52 2019

@author: Administrator
"""

import cv2

facedetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
id = input("Enter Id")
sNo = 0

while(True):
    rect, img=cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facedetector.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sNo = sNo+1
        cv2.imwrite("data/user."+str(id)+"."+str(sNo)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
    cv2.imshow("faces",img)
    cv2.waitKey(1)
    if sNo>100:
        break
cam.release()
cv2.destroyAllWindows()