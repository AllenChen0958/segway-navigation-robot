import cv2
import numpy as np

img = cv2.imread('img.jpg')
template = cv2.imread('template.jpg')

method = cv2.TM_CCOEFF
res = cv2.matchTemplate(img, template, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
w, h, c = template.shape[::-1]
img = img.copy()
# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

print(top_left)
print(bottom_right)
cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)
cv2.imwrite('result.png', img)
