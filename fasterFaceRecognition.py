from Data_Parsing import FORM_FACE_BOX
import cv2
import numpy as np
import face_recognition as fr
import pickle 

myCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = 640
height = 360

myCam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
myCam.set(cv2.CAP_PROP_FPS, 30)
myCam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

faceBoxes = FORM_FACE_BOX.FORM_FACE_BOX()



while True:
    ignore, frame = myCam.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultantBoxes = faceBoxes.FORM_ARRAY(frameRGB)
    print(resultantBoxes)
    for face in resultantBoxes:
        cv2.rectangle(frame, face[0], face[1], (0,0,255), 3)
    cv2.imshow("myFrame", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

myCam.release()