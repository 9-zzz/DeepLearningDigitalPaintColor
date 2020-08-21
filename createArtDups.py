import cv2

im = cv2.imread('./testart/art/bul.jpeg')

for i in range(99):
    cv2.imwrite(f'./testart/art/phpho{i+2}.png',im)