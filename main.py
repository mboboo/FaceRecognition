# import pathlib module to manipulate paths
import pathlib
# import OpenCV module
import cv2

# import DetectedFace class
from DetectFace import DetectedFace

from GraphicalUserInterface import GUI


def main():
    gui = GUI()
    gui.Run()


if __name__ == "__main__":
    main()