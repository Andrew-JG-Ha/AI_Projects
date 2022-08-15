"""
This program uses OpenCV, Mediapipe, Face_Recognition and supplementary libraries
to attempt and improve the Frames Per Second (FPS) of the output video. The
program was able to average 2.42 FPS with 1 person in the frame and average 1.21 FPS
with 2 people in the frame

Differences from baseline (only using Face_Recognition methods):
                    Baseline  | This Program | Difference
1 Person in Frame:  1.70 FPS  | 2.42 FPS     | +0.72 FPS
2 People in Frame:  1.02 FPS  | 1.21 FPS     | +0.19 FPS

The method employed in this program was only slightly better than the baseline methods
"""

from asyncio.windows_events import NULL
from Data_Parsing import FORM_FACE_BOX
import cv2
import face_recognition as FR
import pickle
import os
import time

font = cv2.FONT_HERSHEY_SIMPLEX

myCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = 640
height = 360
tolerance = .6

myCam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
myCam.set(cv2.CAP_PROP_FPS, 50)
myCam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

faceBoxes = FORM_FACE_BOX.FORM_FACE_BOX(minimum_threshold=0.25)

faceEncodings = []
faceNames = []

trainResult = NULL
while trainResult == NULL:
    trainResult = input("Enter 'T' to Train or 'R' to Recognize from a Folder Location (Press Enter for Data From Previous Session): ") 

if trainResult != '':
    encodings_File =[]

    if trainResult == 't' or trainResult == 'T':
        numberOfFaces = input("How many people would you like to recognize: ")
        for person in range(0, int(numberOfFaces)):
            selection1 = input("Enter '1' to Take a Picture or '2' to Access a Known Image: ")
            capturedFrame = []
            if selection1 == '1':
                while True:      
                    ignore, frameCapture = myCam.read()      
                    print("Press 'T' to take a photo")
                    cv2.imshow("myCam", frameCapture)
                    if (cv2.waitKey(1) & 0xff == ord('t')):
                        selection2 = input("Would you like to use this photo? (Y/N): ")
                        if (selection2 == 'y' or selection2 == 'Y'):
                            nameOfPerson = input("Please name this person: ")
                            faceFileName = nameOfPerson+".png"
                            dirname = "faceRecognitionData/trainingData/"
                            faceEncodings.append(FR.face_encodings(frameCapture)[0])
                            faceNames.append(nameOfPerson)
                            cv2.imwrite(os.path.join(dirname, faceFileName), frameCapture)
                            cv2.destroyAllWindows()
                            break
                        else:
                            continue
            elif selection1 == '2':
                filePath = input("Please Enter the Path to this Person's Image: ")
                trainingFace = FR.load_image_file(filePath)
                personFaceEncoding = FR.face_encodings(trainingFace)
                nameOfPerson = input("Please name this person: ")
                faceEncodings.append(personFaceEncoding[0])
                faceNames.append(nameOfPerson)
        
    if trainResult == 'r' or trainResult == 'R':
        filePath = input("Please Enter the Filepath to the Images Folder (Press Enter for Default Filepath): ")
        if filePath == '':
            filePath = "faceRecognitionData\defaultData"
        for root, dirs, files in os.walk(filePath):
            for file in files:
                fullFilePath = os.path.join(root,file)
                personFace = FR.load_image_file(fullFilePath)
                faceLocation = FR.face_locations(personFace)[0]
                personFaceEncode = FR.face_encodings(personFace)[0]
                name = os.path.splitext(file)[0]
                faceEncodings.append(personFaceEncode)
                faceNames.append(name)

    encodings_File.append(faceNames)
    encodings_File.append(faceEncodings)

    with open("faceRecognitionData\encodings_File.pkl", 'wb') as file1:
        pickle.dump(encodings_File, file1)

with open("faceRecognitionData\encodings_File.pkl", 'rb') as file2:
    knownFaces = pickle.load(file2)

# Method containing Mediapipe
while True:
    startTime = time.time()
    ignore, frame = myCam.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    unknownFaces = faceBoxes.FORM_ARRAY(frameRGB)
    for unknownFace in unknownFaces:
        testArray = [(unknownFace[0][1], unknownFace[1][0], unknownFace[1][1], unknownFace[0][0])]
        unknownEncoding = FR.face_encodings(frameRGB, testArray)

        cv2.rectangle(frame, unknownFace[0], unknownFace[1], (255,0,0), 3)
        name = "Unknown"
        if (unknownEncoding != NULL):
            matches = FR.api.compare_faces(knownFaces[1], unknownEncoding[0])
            if (True in matches):
                matchIndex = matches.index(True)
                name = knownFaces[0][matchIndex]
            cv2.putText(frame, name, unknownFace[0], font, 2, (255, 0, 0), 2)
    endTime = time.time()
    deltaTime = endTime - startTime
    framesPerSecond = round(1/deltaTime,3)
    cv2.putText(frame, "FPS: "+str(framesPerSecond), (100,100), font, 1, (0,0,0), 2)
    cv2.imshow("myFrame", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


# """
# The below code is running facial recognition using just the Face_Recognition library methods, 
# the program averaged 1.7 FPS when having 1 person in the frame and 1.02 FPS when there were 2 people in
# the frame
# """

    # Method solely based off Face_Recognition Library
    # startTime = time.time()
    # ignore, frame = myCam.read()
    # unknownFace = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # faceLocations = FR.face_locations(unknownFace)
    # unknownEncodings = FR.face_encodings(unknownFace, faceLocations)

    # for faceLocation, unknownEncoding in zip(faceLocations, unknownEncodings):
    #     top, right, bottom, left = faceLocation
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 3) 
    #     name = 'Unknown Person'
    #     matches = FR.api.compare_faces(knownFaces[1], unknownEncoding)
    #     if (True in matches):
    #         matchIndex = matches.index(True)
    #         name = knownFaces[0][matchIndex]
    #     cv2.putText(frame, name, (left, top), font, 2, (255, 0, 0), 2)

    # endTime = time.time()
    # deltaTime = endTime-startTime
    # framesPerSecond = round(1/deltaTime, 3)
    # cv2.putText(frame, "FPS: "+str(framesPerSecond), (100,100), font, 1, (0,0,0), 2)
    # cv2.imshow('My Faces', frame)
    # if (cv2.waitKey(1) & 0xff == ord('q')):
    #     break

myCam.release()