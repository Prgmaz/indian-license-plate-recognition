import numpy as np
import argparse
import time
import cv2
import os
from PIL import Image
import easyocr
import matplotlib.pyplot as plt

def yolo_license(image, CONFIDENCE_THRESHOLD=0.5, THRESHOLD=0.5):
    weightsPath = os.path.sep.join(["yolo-coco", "yolov3-tiny_last.weights"])
    configPath = os.path.sep.join(["yolo-coco", "yolov3-tiny.cfg"])
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (W, H),
                                swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print(" {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > CONFIDENCE_THRESHOLD:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                            THRESHOLD)

    qualified_boxes = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            qualified_boxes.append(boxes[i])
    
    return qualified_boxes

def recognize_text_license_plate(image, display=True, gpu=False, languages=['en']):
    reader = easyocr.Reader(languages, gpu=gpu)
    w, h, d = image.shape
    print("Shape of image:", image.shape)
    div1 = w // 32
    div2 = h // 32
    w = int(32 * div1)
    h = int(32 * div2)
    image = cv2.resize(image, (h, w))
    outputs = yolo_license(image)
    
    x, y, w, h = w, h, 0, 0
    for o in outputs:
        x_, y_, w_, h_ = o
        x = abs(min(x, x_) )
        y = abs(min(y, y_))
        w = abs(max(w, x+w_))
        h = abs(max(h, y+h_))
    print((x, y, w, h))
    temp = image[y: h, x: w]
    if display and temp.any():
        plt.imshow(temp)
        plt.show()
        plt.imsave("temp-license.png", temp)
        results = reader.recognize("temp-license.png", paragraph=True, detail=0)
        total_text = "".join(results)
        return total_text
    elif not temp.any():
        results = reader.recognize(image, paragraph=True, detail=0)
        total_text = "".join(results)
        return total_text
    return None
