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
    # image = cv2.resize(image, (1280, 720))
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # image = cv2.imread('output/'+str(args['image']).split('.')[0]+'_predictions.jpg')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # gray = cv2.medianBlur(gray, 3)
    # gray = remove_noise_and_smooth(gray)
    # filename = 'output/'+str(args['image']).split('.')[0]+"_preprocessed.jpg"
    # cv2.imwrite(filename, gray)
    # text = pytesseract.image_to_string(Image.open(filename))
    # print(text)
    # cv2.imshow("cropped", image)
    # cv2.imshow("cropped_preprocessed", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
    
    texts = []
    for o in outputs:
        x, y, w, h = o
        temp = image[y: y+h, x: x+w]
        plt.imsave("temp-license.png", temp)
        results = reader.readtext("temp-license.png", paragraph=True, x_ths=2.0, y_ths=1)
        total_text = ""
        for r in results:
            _, text = r
            total_text += text
        if display:
            plt.imshow(temp)
            plt.title(total_text)
            plt.show()
        texts.append(total_text)
    return texts
