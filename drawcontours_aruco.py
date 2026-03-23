import cv2
import cv2.aruco as aruco
import numpy as np
from find_aruco_marker import findAruco

def drawArucoContours(corners, ids, img):
    if ids is not None:
        for i, id in enumerate(ids):
            markerCorners = corners[i][0]

            topleft = (markerCorners[0][0], markerCorners[0][1])
            topright = (markerCorners[1][0], markerCorners[1][1])
            bottomleft = (markerCorners[2][0], markerCorners[2][1])
            bottomright = (markerCorners[3][0], markerCorners[3][1])

            border = np.array([topleft, topright, bottomleft, bottomright], dtype=np.int32)

            cv2.drawContours(img, [border], -1, (0,255,0), 2)
    
    else:
        return


def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()

        if not success:
            break
        
        aruco = findAruco(img, draw=False)
        
        drawArucoContours(aruco[0], aruco[1], img)

        cv2.imshow("cam", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
