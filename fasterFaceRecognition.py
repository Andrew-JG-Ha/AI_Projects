from Data_Parsing import FORM_FACE_BOX
import cv2
import numpy as np
import face_recognition as fr



myCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = 640
height = 360

myCam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
myCam.set(cv2.CAP_PROP_FPS, 30)
myCam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

