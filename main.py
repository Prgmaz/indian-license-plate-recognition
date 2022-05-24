import cv2
import numpy as np
from object_detection import recognize_text_license_plate

if __name__ == "__main__":
    cam = True # True for camera input, False for video input
    cap = None
    if cam:
        cap = cv2.VideoCapture(0)
    else:
        file_path = "" # Enter File Path here 
        cap = cv2.VideoCapture(file_path)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (640, 360))
            cv2.imshow('Frame',frame)
            texts = recognize_text_license_plate(frame)
            print(texts)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()

    cv2.destroyAllWindows()