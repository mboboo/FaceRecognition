# import pathlib module to manipulate paths
import pathlib
# import OpenCV module
import cv2

# import DetectedFace class
from DetectFace import DetectedFace


def main():
    # get the path for Haar cascade
    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
    # load OpenCV detector ( Haar classifier is slow, but accurate )
    classifier = cv2.CascadeClassifier(str(cascade_path))
    # open default camera
    camera = cv2.VideoCapture(0)
    # check if there is a camera
    if camera.isOpened():
        # create object to detect faces
        detectFace = DetectedFace(cascade_path, classifier, camera)
        # run the while loop
        detectFace.run()
    else:
        # if there is no default camera an exception is raised
        raise Exception("[Camera must be open]")


if __name__ == "__main__":
    main()