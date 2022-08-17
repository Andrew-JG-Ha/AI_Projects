"""
Allows the user to play a ping pong game using their fingers. This program tracks the user(s) index fingers
and relates the position to the program where the user can control the paddles in the video stream.

"""


from operator import index
import random
from cv2 import cvtColor
from Data_Parsing import FORM_HANDS_ARRAY
import cv2
import numpy as np

direction = [-1, 1]

player1Score = 0
player2Score = 0

height = 360
width = 640
paddleLength = 60
paddleColor1 = (0,255,0)
paddleColor2 = (0,0,255)

yPosOld = height/2

myCam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
myCam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
myCam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
myCam.set(cv2.CAP_PROP_FPS, 30)
myCam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

myHandsLoc = FORM_HANDS_ARRAY.FORM_HANDS_ARRAY(handsCount=2, width=width, height=height)
player1PaddleY1Loc = 0
player1PaddleY2Loc = 0
player2PaddleY1Loc = 0
player2PaddleY2Loc = 0

ballXPos = int(width/2)
ballYPos = int(height/2)
ballDirectionX = random.choice(direction)
ballDirectionY = random.choice(direction)
ballSpeed = 7
ballRadius = 9
ballColor = (255,0,255)

while True:
    success, frame = myCam.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    myHands, handTypes = myHandsLoc.FORM_ARRAY(frameRGB)

    for hand in myHands:
        indexPos = hand[8]
        xPos = indexPos[0]
        yPos = int(indexPos[1])
        paddleY1Loc = int(yPos - paddleLength/2)
        paddleY2Loc = int(yPos + paddleLength/2)
        if (xPos <= width/2):
            player1PaddleY1Loc = paddleY1Loc
            player1PaddleY2Loc = paddleY2Loc
        elif (xPos >= width/2):
            player2PaddleY1Loc = paddleY1Loc
            player2PaddleY2Loc = paddleY2Loc

    cv2.rectangle(frame, (int(width*0.025),player1PaddleY1Loc), (int(width*0.005),player1PaddleY2Loc), paddleColor1, -1)
    cv2.rectangle(frame, (int(width*0.975),player2PaddleY1Loc), (int(width*0.995),player2PaddleY2Loc), paddleColor2, -1)

    ballXPos = ballXPos+ballSpeed*ballDirectionX
    ballYPos = ballYPos+ballSpeed*ballDirectionY

    if (ballYPos >= player1PaddleY1Loc and ballYPos <= player1PaddleY2Loc and ballXPos-ballRadius <= int(width*0.025) and ballXPos+ballRadius >= int(width*0.005)):
        ballDirectionX = 1

    if (ballYPos >= player2PaddleY1Loc and ballYPos <= player2PaddleY2Loc and ballXPos+ballRadius >= int(width*0.975) and ballXPos-ballRadius <= int(width*0.995)):
        ballDirectionX = -1

    if (ballYPos <= 0 or ballYPos >= height):
        ballDirectionY = ballDirectionY*-1

    if (ballXPos <= -20):
        ballDirectionX = 1
        ballYPos = int(height/2)
        ballXPos = int(width/2)
        player2Score = player2Score + 1
        print(player2Score)

    elif (ballXPos >= width+40):
        ballYPos = int(height/2)
        ballXPos = int(width/2)
        ballDirectionX = -1
        player1Score = player1Score + 1
        print(player1Score)

    scoreBoardText = str("Player 1: " + str(player1Score) + "  VS  Player 2: " +  str(player2Score))

    cv2.circle(frame, (ballXPos, ballYPos), ballRadius, ballColor, -1)
    cv2.putText(frame, scoreBoardText, (int(width/3), 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
    cv2.imshow('myCam', frame)
    cv2.moveWindow("myCam", 0,0)
    if (cv2.waitKey(1) & 0xff == ord('q')):
        break

myCam.release()