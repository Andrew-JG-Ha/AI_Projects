from ast import main
from asyncio.windows_events import NULL
from Data_Parsing import FORM_FACE_BOX
import cv2
import face_recognition as FR
import pandas as pd
import os #normpath and basepath and listdir

train = True

myCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = 640
height = 360

myCam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
myCam.set(cv2.CAP_PROP_FPS, 30)
myCam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

faceBoxes = FORM_FACE_BOX.FORM_FACE_BOX()

trainResult = NULL
while trainResult == NULL:
    trainResult = input("Enter 'T' to train or 'R' to recognize from an existing dataset: ") 

if trainResult == 't' or trainResult == 'T':
    numberOfFaces = input("How many people would you like to train: ")
    faceEncodings = []
    faceNames = []
    for person in range(0, int(numberOfFaces)):
        selection1 = input("Enter '1' to take a picture or '2' to access a known image: ")
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
    myCam.release()
    test = pd.DataFrame({
                        "names": faceNames,
                        "encodings": faceEncodings
                        })
    print(test["names"])



# if trainResult == 'r' or trainResult == 'R':

# while True:
#     ignore, frame = myCam.read()
#     frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     resultantBoxes = faceBoxes.FORM_ARRAY(frameRGB)
#     print(resultantBoxes)
#     for face in resultantBoxes:
#         cv2.rectangle(frame, face[0], face[1], (0,0,255), 3)
#     cv2.imshow("myFrame", frame)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break

myCam.release()

