import pytesseract
import argparse
import cv2
import os
from PIL import Image
import numpy as np

image = cv2.imread('./data/raw/2AQ7.png')
h,w,_ = image.shape
image = cv2.resize(image,(int(w*3),int(h*3)),interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(gray,3)
gray = cv2.GaussianBlur(gray,(3,3),0)
gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

kernel = np.ones((1,1),np.uint8)
gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


text = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)))


print(text)

cv2.imshow("image",gray)
cv2.waitKey(0)