from asyncio.windows_events import NULL
import cv2
from numpy import inner, true_divide
from Data_Parsing import FORM_HANDS_ARRAY
import math
import numpy as np
import pickle


debug = True
train = True

myCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = 640
height = 360

myCam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
myCam.set(cv2.CAP_PROP_FPS, 30)
myCam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

handDetection = FORM_HANDS_ARRAY.FORM_HANDS_ARRAY(2, width=width, height=height)

handIndexes = [0,2,4,5,8,9,12,13,16,17,20]
fingerTipsArray = [8,12,16,20]
innerJointsArray = [5,9,13,17]

font = cv2.FONT_HERSHEY_SIMPLEX
fontSize = 1
fontThickness = 2
fontColor = (255,0,255)

threshold = 20

def calculateAverage(points=[]):
    arrayLength = len(points)
    summationX = 0
    summationY = 0
    if arrayLength > 0:
        for point in points:
            summationX = summationX + point[0]
            summationY = summationY + point[1]
        averageX = summationX/arrayLength
        averageY = summationY/arrayLength
        return(averageX, averageY)
    else:
        return(NULL)

def calculateDeviation(points=[]):
    averageX, averageY = calculateAverage(points)
    arrayLength = len(points)
    summationX = 0
    summationY = 0
    for point in points:
        summationX = summationX + (point[0] - averageX)**2
        summationY = summationY + (point[1] - averageY)**2
    deviationX = math.sqrt(summationX/arrayLength)
    deviationY = math.sqrt(summationY/arrayLength)
    return(deviationX, deviationY)

def calculateHandDistance(handData):
    mapEachPoint = np.zeros([len(handData), len(handData)], dtype = float)
    normalizingFactor = math.sqrt(math.pow(handData[0][0] - handData[5][0],2) + math.pow(handData[0][1] - handData[5][1], 2))
    for row in range(0, len(handData)):
        for column in range(0, len(handData)):
            mapEachPoint[row][column] = math.sqrt(math.pow(handData[row][0] - handData[column][0], 2) + math.pow(handData[row][1] - handData[column][1], 2))/normalizingFactor
    return(mapEachPoint)
        
def findError(gestureMatrix, unknownMatrix):
    error = 0
    for rowIndx in range(0, len(gestureMatrix)):
        for columnIndx in range(0, len(gestureMatrix)):
            error = error + abs(gestureMatrix[rowIndx][columnIndx] - unknownMatrix[rowIndx][columnIndx])
    return(error)
    
def findMatch(knownGestures, unknownGesture, gestureNames, tolerance):
    errorArray = []
    for indx in range(0, len(gestureNames)):
        error = findError(knownGestures[indx], unknownGesture)
        errorArray.append(error)
    minIndex = 0
    errorMin = errorArray[minIndex]
    for indx in range(0, len(errorArray)):
        if errorArray[indx] < errorMin:
            errorMin = errorArray[indx]
            minIndex = indx
    if errorMin < tolerance:
        gesture = gestureNames[minIndex]
    else:
        gesture = "Unknown"
    return(gesture)

train = int(input("Enter '1' to Train, and '0' to Recognize "))

if train == 1:
    knownGestures = []
    gestureNames = []
    numGestures = int(input("How many gestures would you like to input? "))
    for indx in range(0, numGestures, 1):
        prompt = "Name of Gesture #" + str(indx+1)+': '
        name = input(prompt)
        gestureNames.append(name)
    trainName = input("Filename for training data? (Press Enter for Default) ")
    if trainName == '':
        trainName='handGestureData/default'
    trainName=trainName+'.pkl'

if train == 0:
    trainName = input("What Training Data Do You Want to Use? (Press Enter for Default) ")
    if trainName == '':
        trainName = 'handGestureData/default'
    trainName = trainName + '.pkl'
    with open(trainName, 'rb') as f:
        gestureNames = pickle.load(f)
        knownGestures = pickle.load(f)

while True:
    handID = "Unknown"
    ignore, frame = myCam.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    handArray, handPosition = handDetection.FORM_ARRAY(frameRGB)

    for hand, side in zip(handArray,handPosition):
        fingerTipPositions = []
        innerJointsPositions = []
        keyPointsPositions = []
        
        # Grabbing seperate arrays
        for indx in handIndexes:
            if indx in fingerTipsArray:
                fingerTipPositions.append(hand[indx])
            elif indx in innerJointsArray:
                innerJointsPositions.append(hand[indx])
            keyPointsPositions.append(hand[indx])
            if debug == True:
                cv2.circle(frame, hand[indx], 3, (0,255,0), -1)

        # Calculating average in innerJoints and fingerTips as well as standard deviation
        fingerTipsAverageX, fingerTipsAverageY = calculateAverage(fingerTipPositions)
        innerJointsAverageX, innerJointsAverageY = calculateAverage(innerJointsPositions)

        keyPointsPositions.append((fingerTipsAverageX, fingerTipsAverageY))
        keyPointsPositions.append((innerJointsAverageX, innerJointsAverageY))

        if train == 1:
            if hand != []:
                print("Show gesture: " + gestureNames[len(knownGestures)] + ", press 'T' when ready")
                if cv2.waitKey(1) & 0xff == ord('t'):
                    knownGestureTrain = calculateHandDistance(keyPointsPositions)
                    knownGestures.append(knownGestureTrain)
                    if (len(knownGestures) == len(gestureNames)):
                        train = 0
                        with open(trainName, 'wb') as f:
                            pickle.dump(gestureNames, f)
                            pickle.dump(knownGestures, f)             
                    
        if train == 0:
            if hand != []:
                unknownGesture = calculateHandDistance(keyPointsPositions)
                handID = findMatch(knownGestures, unknownGesture, gestureNames, threshold)             

        if debug == True:
            fingerTipsDeviationX, fingerTipsDeviationY = calculateDeviation(fingerTipPositions)
            innerJointDeviationX, innerJointDeviationY = calculateDeviation(innerJointsPositions)

            negativeDeviationFingerTipsY = int(fingerTipsAverageY - fingerTipsDeviationY)
            positiveDeviationFingerTipsY = int(fingerTipsAverageY + fingerTipsDeviationY)
            negativeDeviationFingerTipsX = int(fingerTipsAverageX - fingerTipsDeviationX)
            positiveDeviationFingerTipsX = int(fingerTipsAverageX + fingerTipsDeviationX)

            deviationCircleColor = (255,255,0)
            cv2.circle(frame, (int(fingerTipsAverageX), int(fingerTipsAverageY)), 5, (0,0,255), -1)
            cv2.circle(frame, (int(innerJointsAverageX), int(innerJointsAverageY)), 5, (255,0,0), -1)
            
            cv2.circle(frame, (negativeDeviationFingerTipsX, int(fingerTipsAverageY)), 5, deviationCircleColor, -1)
            cv2.circle(frame, (positiveDeviationFingerTipsX, int(fingerTipsAverageY)), 5, deviationCircleColor, -1)
            cv2.circle(frame, (int(fingerTipsAverageX), negativeDeviationFingerTipsY), 5, deviationCircleColor, -1)
            cv2.circle(frame, (int(fingerTipsAverageX), positiveDeviationFingerTipsY), 5, deviationCircleColor, -1)

        cv2.putText(frame, handID, (hand[0][0],(hand[0][1]+25)), font, fontSize, fontColor, fontThickness)
    cv2.imshow("myCam", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

myCam.release()