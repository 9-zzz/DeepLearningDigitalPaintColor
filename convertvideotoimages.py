import cv2
import os
import argparse

vid = cv2.VideoCapture('eva.mp4')

os.makedirs('images', exist_ok=True)

idx = 0
while(vid.isOpened()):
    ret, frame = vid.read()
    if ret == True:
        cv2.imwrite(f'images\\{idx}.png',frame)
        idx += 1