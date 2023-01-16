import cv2

class DetectedFace:

    def __init__(self, cascade_path, classifier, camera):
        self.cascade_path = cascade_path
        self.classifier = classifier
        self.camera = camera

    def run(self):
        while True:
            _, frame = self.camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.classifier.detectMultiScale(
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, width, height) in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

            cv2.imshow("Faces", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        self.camera.release()
        cv2.destroyAllWindows()