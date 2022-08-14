import mediapipe as mp

class FORM_POSE_ARRAY:

    def __init__(self, frame_width=640, frame_height=360, staticImage = False ,modelComplexity = 1, segmentation = False, detectionConfidence = 0.5, trackingConfidence = 0.5):
        self.width = frame_width
        self.height = frame_height
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode = staticImage, 
            model_complexity = modelComplexity, 
            enable_segmentation = segmentation, 
            min_detection_confidence = detectionConfidence, 
            min_tracking_confidence = trackingConfidence
            )
    
    def FORM_ARRAY(self, frameRGB):
        """
        Forms an array of 33 elements retaining the X and Y coordinates of the landmarks
        Takes in a frame in RGB and returns an array of 33 elements
        """
        poseArray = []
        results = self.mp_pose.process(frameRGB)
        if results.pose_landmarks != None:
            for landmark in results.pose_landmarks.landmark:
                poseArray.append((int(landmark.x*self.width), int(landmark.y*self.height)))
        return(poseArray)

