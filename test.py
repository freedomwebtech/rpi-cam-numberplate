import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
from paddleocr import PaddleOCR
import numpy as np
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1020,500)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
model=YOLO('best_float32.tflite')
ocr = PaddleOCR()

def perform_ocr(image_array):
    if image_array is None:
        raise ValueError("Image is None")

    # Perform OCR on the image array
    results = ocr.ocr(image_array, rec=True)  # rec=True enables text recognition
    detected_text = []

    # Process OCR results
    if results[0] is not None:
#        print(results)
        for result in results[0]:
            print(result)
            text = result[1][0]
            detected_text.append(text)
      
    # Join all detected texts into a single string
    return ''.join(detected_text)
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count=0
while True:
    im= picam2.capture_array()
    
    count += 1
    if count % 3 != 0:
        continue
    im=cv2.flip(im,-1)
    results=model.predict(im,imgsz=240)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    
    
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        crop = im[y1:y2, x1:x2]
        crop=cv2.resize(crop,(110,70))
        text = perform_ocr(crop)
        text = text.replace('(', '').replace(')', '').replace(',', '').replace(']','').replace('-',' ')
        print(text)
        cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
#        cvzone.putTextRect(im,f'{c}',(x1,y1),1,1)
        cvzone.putTextRect(im, f'{text}', (x1 - 50, y1 - 30), 1, 1)
    cv2.imshow("Camera", im)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()