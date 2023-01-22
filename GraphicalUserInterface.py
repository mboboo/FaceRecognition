import customtkinter
import pathlib

# import DetectedFace class
from DetectFace import DetectedFace

import cv2

from PIL import Image, ImageTk

class GUI():

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")

    root = customtkinter.CTk()
    root.geometry("1200x800")

    frame = customtkinter.CTkFrame(master=root)
    frame.pack(pady=20, padx=60, fill="both", expand=True)

    cameraLabel = customtkinter.CTkLabel(master=frame)
    cameraLabel.pack()
    
    def Run(self):
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
            while True:
                # wait for input ('q') to close the window
                if cv2.waitKey(1) == ord("q"):
                    break

                # get last frame from camera
                cv2LastFrame = detectFace.DetectFaceInLatestFrame()
                # convert to PIL.Image
                img = Image.fromarray(cv2LastFrame)
                # convert to Tkinter image format
                imageTk = ImageTk.PhotoImage(image=img)
                # display last frame
                self.cameraLabel.configure(image=imageTk)
                self.root.mainloop()
        else:
            # if there is no default camera an exception is raised
            raise Exception("[Camera must be open]")

        
       
        # close capturing device
        self.camera.release()
        # close all windows
        cv2.destroyAllWindows()