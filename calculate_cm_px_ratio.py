import cv2
from find_aruco_marker import findAruco
from drawcontours_aruco import drawArucoContours
import math


actualSideLength = 4.7 #cm

def calculateLength(id, corner):
    '''
    calculates side length of detected aruco marker in pixels
    '''


    pts = corner[0]

    dx = pts[0][0] - pts[1][0]
    dy = pts[0][1] - pts[1][1]

    l = math.sqrt(dx ** 2 + dy ** 2)

    return l

def calculateRatio(actualLength, apparentLength):
    '''
    calculates the ratio of actual length (cm) to apparent length (px) of the aruco marker
    '''
    return actualLength/apparentLength


def main():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        
        if not success:
            break
        
        [corners, ids] = findAruco(img, draw=False)

        apparentSideLength = 0
        ratio = 0

        if len(corners) != 0:
            for corner, id in zip(corners, ids):
                marker_id = int(id.item())
                if marker_id == 5:
                    drawArucoContours([corner], [marker_id], img)
                    apparentSideLength = calculateLength(marker_id, corner)
                    ratio = calculateRatio(actualSideLength, apparentSideLength)

        cv2.putText(img, f"{apparentSideLength: .2f} px", (10, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.putText(img, f"cm to pixel ratio: {ratio: .2f}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)
        # cv2.putText(img, f"{apparentSideLength * ratio:.2f}", (int(corner[0][0][0]), int(corner[0][0][1])), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)

        cv2.imshow("capture", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()