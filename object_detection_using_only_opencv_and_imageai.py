from imageai.Detection import ObjectDetection
import cv2
import os
import time
import numpy as np
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel()
cap = cv2.VideoCapture(0)
time.sleep(3)
while(cap.isOpened()):
    flag, img = cap.read()
    img = np.flip(img,axis=1)
    if not flag:
        break
    else:
        cv2.imwrite("new_image.jpg",img)
        detection = detector.detectObjectsFromImage(input_image="new_image.jpg",output_image_path="new_image.jpg")
        filename = cv2.imread("new_image.jpg")
        cv2.imshow("window",filename)
        k = cv2.waitKey(1)
        if k==27:
            break
        else:
            continue
cap.release()
os.remove("new_image.jpg")
cv2.destroyAllWindows()