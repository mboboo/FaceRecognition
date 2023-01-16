import pathlib
import cv2

from DetectFace import DetectedFace


def main():
    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(str(cascade_path))
    camera = cv2.VideoCapture(0)
    if camera.isOpened():
        detectFace = DetectedFace(cascade_path, classifier, camera)
        detectFace.run()
    else:
        raise Exception("[Camera must be open]")


if __name__ == "__main__":
    main()