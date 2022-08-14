import mediapipe as mp

class FORM_HANDS_ARRAY:

    def __init__(self, handsCount=2, width=640, height=360):
        self.width = width
        self.height = height
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands = handsCount, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def FORM_ARRAY(self, frame):
        handsPositions = []
        handTypes = []
        results = self.mp_hands.process(frame)
        if (results.multi_hand_landmarks != None):
            for types in results.multi_handedness:
                handType = types.classification[0].label
                handTypes.append(handType)
            for handLandmarks in results.multi_hand_landmarks:
                hand = []
                for landmark in handLandmarks.landmark:
                    hand.append((int(landmark.x*self.width), int(landmark.y*self.height)))
                handsPositions.append(hand)
        return(handsPositions, handTypes)

    