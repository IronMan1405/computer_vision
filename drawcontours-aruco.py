import cv2
import cv2.aruco as aruco
import numpy as np

def findAruco(img, size=6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    arucoParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParams)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        if draw:
            aruco.drawDetectedMarkers(img, corners, ids)
    else:
        cv2.putText(img, "no marker detected", (10, 100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (0, 255, 0), 2)

    return [corners, ids]

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
