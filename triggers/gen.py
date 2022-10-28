import cv2
import numpy as np


"""
trigger = np.zeros((11, 11, 3), dtype=np.uint8)

for i in range(11):
    for j in range(11):
        if (i+j) % 2 == 0:
            trigger[i][j][:] = 0
        else:
            trigger[i][j][:] = 255

cv2.imwrite('badnet_11x11.png', trigger)"""


trigger = cv2.imread('badnet_patch.png')
trigger = cv2.resize(trigger, (21,21), cv2.INTER_NEAREST_EXACT)

cv2.imwrite('badnet_high_res.png', trigger)
