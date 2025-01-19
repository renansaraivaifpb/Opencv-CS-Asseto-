'''
pip install numpy
pip install Pillow
pip install opencv-python
'''

import numpy as np
from PIL import ImageGrab
import cv2

import time

def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1 = 200, threshold2 = 300)
    return processed_img

while(True):
    screen = np.array(ImageGrab.grab(bbox=(0,40,720,400)))
    new_screen = process_img(screen)
    cv2.imshow('window', new_screen)
    #cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
