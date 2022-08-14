import mediapipe as mp 

class FORM_FACEMESH_ARRAY:
    def __init__(self, _static_image_mode = False, _max_num_faces = 3, _refine_landmarks = True, _min_detection_confidence = 0.5, _width = 640, _height=360):
        self.width = _width
        self.height = _height
        self.faceMesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode = _static_image_mode, 
            max_num_faces = _max_num_faces, 
            refine_landmarks = _refine_landmarks, 
            min_detection_confidence = _min_detection_confidence
            )
        self.faceMeshArray = []
    
    def FORM_ARRAY(self, frameRGB):
        faceArray = []
        results = self.faceMesh.process(frameRGB)
        if results.multi_face_landmarks != None:
            for face in results.multi_face_landmarks:
                landmarkTupleArray = []
                for landmark in face.landmark:
                    xCoord = int(landmark.x*self.width)
                    yCoord = int(landmark.y*self.height)
                    landmarkTupleArray.append((xCoord,yCoord))
                faceArray.append(landmarkTupleArray)
        self.faceMeshArray = faceArray
        return(faceArray)

    def GET_IRISES(self):
        faceOutlineArray = []
        if (len(self.faceMeshArray) == 0):
            print("Please Run FORM_ARRAY Method to retrive data")
            return(self.faceMeshArray)
        else:
            for face in self.faceMeshArray:
                faceOutlineArray.append(face[len(face)-10:])
            return(faceOutlineArray)

    # def GET_EYES(self):

    # def GET_MOUTH(self):

    def GET_FACE_OUTLINE(self):
        faceOutlineArray = []
        faceOutlineIndex = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338]
        3,4,1
        if (len(self.faceMeshArray) == 0):
            print("Please Run FORM_ARRAY Method to retrive data")
            return(self.faceMeshArray)
        else:
            for face in self.faceMeshArray:
                faceLandmarkArray = []
                for indx in faceOutlineIndex:
                    faceLandmarkArray.append(face[indx])
                faceOutlineArray.append(faceLandmarkArray)
        return(faceOutlineArray)

    def GET_LEFT_FACE(self):
        faceOutlineArray = []
        if (len(self.faceMeshArray) == 0):
            print("Please Run FORM_ARRAY Method to retrive data")
            return(self.faceMeshArray)
        else:
            for face in self.faceMeshArray:
                faceOutlineArray.append(face[:247])
            return(faceOutlineArray)

    def GET_RIGHT_FACE(self):
        faceOutlineArray = []
        if (len(self.faceMeshArray) == 0):
            print("Please Run FORM_ARRAY Method to retrive data")
            return(self.faceMeshArray)
        else:
            for face in self.faceMeshArray:
                faceOutlineArray.append(face[248:len(face)-11])
            return(faceOutlineArray)