import cv2

# class to detect face using OpenCV
class DetectedFace:

    # parameterized constructor
    def __init__(self, cascade_path, classifier, camera):
        self.cascade_path = cascade_path
        self.classifier = classifier
        self.camera = camera

    # function that runs in a loop unless no default camera is detected
    def DetectFaceInLatestFrame(self):
        # get every frame from the camera
        _, frame = self.camera.read()
        # convert every frame to gray image ( OpenCV expects gray images)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in a list of faces
        faces = self.classifier.detectMultiScale(
            # matrix containing an image where objects are detected
            grayFrame,
            # some faces may be closer to the camera
            scaleFactor = 1.2,
            # how many neighbors each candidate rectangle should have to retain it
            minNeighbors = 5,
            # objects smaller than the specified size will be ignored
            minSize = (30, 30),
            # old format cascade
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # draw rectangle around every face detected
        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

        return frame