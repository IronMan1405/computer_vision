import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt

arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

id = int(input("Enter marker id: "))
size = int(input("Enter size in pixels: "))

marker = aruco.generateImageMarker(arucoDict, id, size)

print(marker.dtype, marker.shape)

pad = size // 8
padded = cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255,255,255))
cv2.imwrite(f"markers/marker_{id}_{size}.png", padded)

while True:
    cv2.imshow("marker", marker)
    cv2.imshow("padded", padded)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
