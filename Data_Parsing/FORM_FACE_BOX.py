import mediapipe as mp

class FORM_FACE_BOX:
    def __init__(self, _model_selection = 0, minimum_threshold = 0.5, width = 640, height = 360):
        self.width = width
        self.height = height
        self.boxFace = mp.solutions.face_detection.FaceDetection(model_selection = _model_selection, min_detection_confidence = minimum_threshold)
    
    def FORM_ARRAY(self, frameRGB):
        faces = []
        results = self.boxFace.process(frameRGB)
        if results.detections != None:
            for face in results.detections:
                boundingBox = face.location_data.relative_bounding_box
                topLeftCorner = (int(boundingBox.xmin*self.width),int(boundingBox.ymin*self.height))
                bottomRightCorner = (int((boundingBox.xmin+boundingBox.width)*self.width), int((boundingBox.ymin+boundingBox.height)*self.height))
                faces.append([topLeftCorner, bottomRightCorner])
        return(faces)





