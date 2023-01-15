import pathlib
import cv2


class DetectFace:

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
                scaleFactor = 1.3,
                minNeighbors = 3,
                minSize = (30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, width, height) in faces:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

            cv2.imshow("Faces", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        self.camera.release()
        cv2.destroyAllWindows()


def main():
    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(str(cascade_path))
    camera = cv2.VideoCapture(0)

    detectFace = DetectFace(cascade_path, classifier, camera)
    detectFace.run()

if __name__ == "__main__":
    main()