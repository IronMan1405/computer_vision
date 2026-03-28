import cv2
import numpy as np
from find_aruco_marker import findAruco
from drawcontours_aruco import drawArucoContours
import math

def main():
    cap = cv2.VideoCapture(0)

    cnr1 = None
    cnr2 = None

    while True:
        success, img = cap.read()

        if not success:
            break
        
        [corners, ids] = findAruco(img, draw=False)

        workspace_corners = []
        if ids is not None:
            for cnr, id in zip(corners, ids):
                marker_id = int(id.item())
                if marker_id == 42:
                    cnr1 = tuple(map(int, cnr[0][1]))
                if marker_id == 249:
                    cnr2 = tuple(map(int, cnr[0][0]))

        if cnr1 is not None:
            cv2.circle(img, cnr1, 2, (0,0,127), 2)
        if cnr2 is not None:
            cv2.circle(img, cnr2, 2, (0,0,127), 2)

        if cnr1 is not None and cnr2 is not None:
            cv2.rectangle(img, cnr1, cnr2, 255, 2)

            r = cv2.arcLength(np.array([cnr1, cnr2]), False)
            cv2.circle(img, cnr2, int(r), (0,127,0), 2)

        cv2.imshow("cam", img)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()