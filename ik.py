import cv2
import cv2.aruco as aruco
import numpy as np

def augmentAruco(corners, ids, img, augImg):
    corners = corners[0]  # shape: (4, 2)

    topleft     = corners[0][0], corners[0][1]
    topright    = corners[1][0], corners[1][1]
    bottomright = corners[2][0], corners[2][1]
    bottomleft  = corners[3][0], corners[3][1]

    border = np.array([topleft, topright, bottomright, bottomleft], dtype=np.float32)

    h, w = img.shape[:2]
    augH, augW = augImg.shape[:2]

    # Map augImg corners to marker corners in the frame
    imgpts = np.array([[0, 0], [augW, 0], [augW, augH], [0, augH]], dtype=np.float32)

    matrix, _ = cv2.findHomography(imgpts, border)  # note: imgpts -> border (not reversed)

    # Warp augImg into the frame's perspective
    imgOut = cv2.warpPerspective(augImg, matrix, (w, h))

    # Create a mask from the warped region and blend onto img
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, border.astype(np.int32), 255)

    # Black out the marker region in img, then add warped image
    img[mask == 255] = 0
    img = cv2.add(img, imgOut * (mask[:, :, np.newaxis] // 255))

    return img
